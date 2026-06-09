//! Silero VAD as a standalone Path 3 loadable plugin.
//!
//! This crate is fully decoupled from `remotemedia-core` — it depends
//! only on `remotemedia-plugin-sdk` for the skinny trait/type surfaces
//! and on `ort` for ONNX Runtime inference (pinned in this crate's own
//! `Cargo.toml` so the plugin's tree never unifies with the host's).
//!
//! ## Node types exported
//!
//!   SileroVADNode             — Audio → emits Json VAD event + audio passthrough
//!   SpeculativeVADCoordinator — Audio → emits audio (speculative) + Json VAD event
//!                                       + optional CancelSpeculation ControlMessage
//!
//! Both nodes use the multi-output FFI path (`FfiNode::process_multi`)
//! so the JSON event AND the audio (and any control messages) cross the
//! dlopen boundary intact.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex as PLMutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{Mutex as TokioMutex, OnceCell};

use remotemedia_plugin_sdk::abi_stable::sabi_trait::TD_Opaque;
use remotemedia_plugin_sdk::abi_stable::std_types::{ROk, RResult, RString};
use remotemedia_plugin_sdk::adapter::StreamingNodeFfiAdapter;
use remotemedia_plugin_sdk::traits::streaming::AsyncStreamingNode;
use remotemedia_plugin_sdk::traits::VoiceActivityDetectorBackend;
use remotemedia_plugin_sdk::types::{AudioSamples, ControlMessageType, Error, RuntimeData};
use remotemedia_plugin_sdk::{FfiNodeBox, FfiNodeFactory, FfiNode_TO};

use ort::{
    execution_providers::CPUExecutionProvider,
    session::{Session, SessionOutputs},
    value::Tensor,
};

extern "C" {
    fn silero_android_force_libcxx_streams();
}

// ---------------------------------------------------------------------------
// SileroVADNode config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SileroVADConfig {
    #[serde(alias = "modelPath")]
    pub model_path: Option<String>,
    pub threshold: f32,
    #[serde(alias = "negThreshold")]
    pub neg_threshold: Option<f32>,
    #[serde(alias = "samplingRate")]
    pub sampling_rate: u32,
    #[serde(alias = "minSpeechDurationMs")]
    pub min_speech_duration_ms: u32,
    #[serde(alias = "minSilenceDurationMs")]
    pub min_silence_duration_ms: u32,
    #[serde(alias = "speechPadMs")]
    pub speech_pad_ms: u32,
}

impl Default for SileroVADConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            threshold: 0.5,
            neg_threshold: None,
            sampling_rate: 16_000,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-session VAD state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct VADState {
    triggered: bool,
    temp_end_samples: usize,
    current_sample: usize,
    /// Silero VAD combined hidden state — `[2, 1, 128]` = 256 floats.
    state: Vec<f32>,
}

impl Default for VADState {
    fn default() -> Self {
        Self {
            triggered: false,
            temp_end_samples: 0,
            current_sample: 0,
            state: vec![0.0; 2 * 128],
        }
    }
}

// ---------------------------------------------------------------------------
// SileroVADNode
// ---------------------------------------------------------------------------

pub struct SileroVADNode {
    config: SileroVADConfig,
    session: OnceCell<Arc<TokioMutex<Session>>>,
    states: Arc<TokioMutex<HashMap<String, VADState>>>,
}

impl SileroVADNode {
    pub fn new(config: SileroVADConfig) -> Self {
        unsafe { silero_android_force_libcxx_streams() };
        Self {
            config,
            session: OnceCell::new(),
            states: Arc::new(TokioMutex::new(HashMap::new())),
        }
    }

    fn effective_neg_threshold(&self) -> f32 {
        self.config
            .neg_threshold
            .unwrap_or((self.config.threshold - 0.15).max(0.0))
            .min(self.config.threshold)
    }

    async fn get_or_init_session(&self) -> Result<&Arc<TokioMutex<Session>>, Error> {
        self.session
            .get_or_try_init(|| async {
                tracing::info!("Initializing Silero VAD ONNX model");

                let model_path = self
                    .config
                    .model_path
                    .as_deref()
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|| std::path::PathBuf::from("silero_vad.onnx"));
                if !model_path.exists() {
                    tracing::info!("Downloading Silero VAD model...");
                    let url = "https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx";
                    let client = reqwest::Client::builder()
                        .user_agent("remotemedia-silero-vad-plugin/0.3")
                        .build()
                        .map_err(|e| Error::Execution(format!("HTTP client: {e}")))?;
                    let response = client
                        .get(url)
                        .send()
                        .await
                        .map_err(|e| Error::Execution(format!("download: {e}")))?;
                    if !response.status().is_success() {
                        return Err(Error::Execution(format!(
                            "download failed: HTTP {}",
                            response.status()
                        )));
                    }
                    let bytes = response
                        .bytes()
                        .await
                        .map_err(|e| Error::Execution(format!("read body: {e}")))?;
                    tokio::fs::write(&model_path, &bytes)
                        .await
                        .map_err(|e| Error::Execution(format!("save model: {e}")))?;
                    tracing::info!("Silero VAD model downloaded ({} bytes)", bytes.len());
                }

                let session = Session::builder()
                    .map_err(|e| Error::Execution(format!("ort builder: {e}")))?
                    .with_execution_providers([CPUExecutionProvider::default().build()])
                    .map_err(|e| Error::Execution(format!("ort EP: {e}")))?
                    .commit_from_file(&model_path)
                    .map_err(|e| Error::Execution(format!("ort load: {e}")))?;

                tracing::info!("Silero VAD model loaded");
                Ok(Arc::new(TokioMutex::new(session)))
            })
            .await
    }

    async fn run_vad(&self, audio: &[f32], state: &mut VADState) -> Result<f32, Error> {
        let session_arc = self.get_or_init_session().await?;
        let mut session = session_arc.lock().await;

        let chunk_size = audio.len();

        let input_tensor = Tensor::from_array(([1, chunk_size], audio.to_vec()))
            .map_err(|e| Error::Execution(format!("ort input tensor: {e}")))?;
        let state_tensor = Tensor::from_array(([2, 1, 128], state.state.clone()))
            .map_err(|e| Error::Execution(format!("ort state tensor: {e}")))?;
        let sr_tensor = Tensor::from_array(([0usize; 0], vec![self.config.sampling_rate as i64]))
            .map_err(|e| Error::Execution(format!("ort sr tensor: {e}")))?;

        let outputs: SessionOutputs = session
            .run(ort::inputs![
                "input" => input_tensor,
                "state" => state_tensor,
                "sr" => sr_tensor,
            ])
            .map_err(|e| Error::Execution(format!("ort run: {e}")))?;

        let (_, output_data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Execution(format!("ort extract output: {e}")))?;
        let speech_prob = output_data[0];

        let (_, state_data) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Execution(format!("ort extract stateN: {e}")))?;
        state.state.copy_from_slice(state_data);

        Ok(speech_prob)
    }

    fn resample_audio(&self, audio: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
        if from_sr == to_sr {
            return audio.to_vec();
        }
        let ratio = from_sr as f32 / to_sr as f32;
        let new_len = (audio.len() as f32 / ratio) as usize;
        (0..new_len)
            .map(|i| {
                let pos = i as f32 * ratio;
                let idx = pos as usize;
                let frac = pos - idx as f32;
                if idx + 1 < audio.len() {
                    audio[idx] * (1.0 - frac) + audio[idx + 1] * frac
                } else {
                    audio[idx]
                }
            })
            .collect()
    }

    async fn vad_event_for(
        &self,
        mono: Vec<f32>,
        session_id: Option<&str>,
    ) -> Result<Value, Error> {
        let key = session_id.unwrap_or("default").to_string();
        let mut states = self.states.lock().await;
        let state = states.entry(key).or_insert_with(VADState::default);

        let (rms, peak) = if mono.is_empty() {
            (0.0_f32, 0.0_f32)
        } else {
            let mut sum_sq = 0.0_f64;
            let mut pk = 0.0_f32;
            for &s in &mono {
                sum_sq += (s as f64) * (s as f64);
                let a = s.abs();
                if a > pk {
                    pk = a;
                }
            }
            ((sum_sq / mono.len() as f64).sqrt() as f32, pk)
        };

        let speech_prob = self.run_vad(&mono, state).await?;
        let neg_threshold = self.effective_neg_threshold();

        let mut is_speech_start = false;
        let mut is_speech_end = false;

        if speech_prob >= self.config.threshold {
            if !state.triggered {
                is_speech_start = true;
                state.triggered = true;
                tracing::info!("Speech started (prob={:.3})", speech_prob);
            }
            state.temp_end_samples = 0;
        } else if state.triggered {
            if speech_prob >= neg_threshold {
                state.temp_end_samples = 0;
            } else {
                state.temp_end_samples += mono.len();
                let silence_ms = (state.temp_end_samples as f32 / self.config.sampling_rate as f32
                    * 1000.0) as u32;
                if silence_ms >= self.config.min_silence_duration_ms {
                    is_speech_end = true;
                    state.triggered = false;
                    state.temp_end_samples = 0;
                    tracing::info!("Speech ended (silence={}ms)", silence_ms);
                }
            }
        }

        state.current_sample += mono.len();

        Ok(serde_json::json!({
            "has_speech": speech_prob >= self.config.threshold,
            "speech_probability": speech_prob,
            "is_speech_start": is_speech_start,
            "is_speech_end": is_speech_end,
            "timestamp_ms": (state.current_sample as f32 / self.config.sampling_rate as f32 * 1000.0) as u64,
            "rms": rms,
            "peak": peak,
            "samples": mono.len(),
            "sample_rate": self.config.sampling_rate,
        }))
    }

    fn extract_mono(&self, data: &RuntimeData) -> Option<Vec<f32>> {
        match data {
            RuntimeData::Audio {
                samples,
                sample_rate,
                channels,
                ..
            } => {
                let resampled = if *sample_rate != self.config.sampling_rate {
                    self.resample_audio(samples.as_slice(), *sample_rate, self.config.sampling_rate)
                } else {
                    samples.as_slice().to_vec()
                };
                let mono = if *channels > 1 {
                    resampled
                        .chunks(*channels as usize)
                        .map(|c| c.iter().sum::<f32>() / *channels as f32)
                        .collect()
                } else {
                    resampled
                };
                Some(mono)
            }
            _ => None,
        }
    }
}

#[async_trait]
impl AsyncStreamingNode for SileroVADNode {
    fn node_type(&self) -> &str {
        "SileroVADNode"
    }

    async fn process(&self, data: RuntimeData) -> Result<RuntimeData, Error> {
        let mono = match self.extract_mono(&data) {
            Some(m) => m,
            None => return Ok(data),
        };
        let event = self.vad_event_for(mono, None).await?;
        Ok(RuntimeData::Json(event))
    }

    /// Multi-output: emit JSON event AND pass through the original
    /// audio (in that order). Matches the in-tree behaviour the
    /// downstream audio-buffer accumulator depends on.
    async fn process_streaming<F>(
        &self,
        data: RuntimeData,
        session_id: Option<String>,
        mut callback: F,
    ) -> Result<usize, Error>
    where
        F: FnMut(RuntimeData) -> Result<(), Error> + Send,
    {
        let mono = match self.extract_mono(&data) {
            Some(m) => m,
            None => {
                callback(data)?;
                return Ok(1);
            }
        };
        let event = self.vad_event_for(mono, session_id.as_deref()).await?;
        callback(RuntimeData::Json(event))?;
        callback(data)?;
        Ok(2)
    }
}

impl VoiceActivityDetectorBackend for SileroVADNode {
    fn reset_buffers(&self, session_id: &str) {
        if let Ok(mut states) = self.states.try_lock() {
            states.insert(session_id.to_string(), VADState::default());
        } else {
            let mut states = self.states.blocking_lock();
            states.insert(session_id.to_string(), VADState::default());
        }
    }

    fn evaluate_veto(&self, session_id: &str, audio_samples: &[f32]) -> Result<bool, Error> {
        let handle = tokio::runtime::Handle::try_current()
            .map_err(|e| Error::Execution(format!("No active tokio runtime thread: {e}")))?;
        handle.block_on(async {
            let event = self
                .vad_event_for(audio_samples.to_vec(), Some(session_id))
                .await?;
            let has_speech = event
                .get("has_speech")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            Ok(has_speech)
        })
    }
}

#[derive(Default)]
pub struct SileroVADNodeFactory;

impl FfiNodeFactory for SileroVADNodeFactory {
    fn node_type(&self) -> RString {
        RString::from("SileroVADNode")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        let cfg: SileroVADConfig = serde_json::from_str(params.as_str()).unwrap_or_default();
        ROk(FfiNode_TO::from_value(
            StreamingNodeFfiAdapter::new(SileroVADNode::new(cfg)),
            TD_Opaque,
        ))
    }
}

// ---------------------------------------------------------------------------
// SpeculativeVADCoordinator
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpeculativeVADCoordinatorConfig {
    pub vad_threshold: f32,
    #[serde(alias = "samplingRate")]
    pub sample_rate: u32,
    #[serde(alias = "minSpeechDurationMs")]
    pub min_speech_duration_ms: u32,
    #[serde(alias = "minSilenceDurationMs")]
    pub min_silence_duration_ms: u32,
    #[serde(alias = "lookbackMs")]
    pub lookback_ms: u32,
    #[serde(alias = "speechPadMs")]
    pub speech_pad_ms: u32,
}

impl Default for SpeculativeVADCoordinatorConfig {
    fn default() -> Self {
        Self {
            vad_threshold: 0.5,
            sample_rate: 16_000,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            lookback_ms: 150,
            speech_pad_ms: 30,
        }
    }
}

#[derive(Debug)]
struct CoordinatorState {
    audio_buffer: VecDeque<f32>,
    buffer_capacity: usize,
    speech_start_sample: Option<usize>,
    current_sample: usize,
    segment_counter: u64,
    speculations_accepted: u64,
    speculations_cancelled: u64,
    speech_triggered: bool,
    silence_samples: usize,
}

impl CoordinatorState {
    fn new(buffer_capacity: usize) -> Self {
        Self {
            audio_buffer: VecDeque::with_capacity(buffer_capacity),
            buffer_capacity,
            speech_start_sample: None,
            current_sample: 0,
            segment_counter: 0,
            speculations_accepted: 0,
            speculations_cancelled: 0,
            speech_triggered: false,
            silence_samples: 0,
        }
    }

    fn acceptance_rate(&self) -> f64 {
        let total = self.speculations_accepted + self.speculations_cancelled;
        if total == 0 {
            return 1.0;
        }
        self.speculations_accepted as f64 / total as f64
    }
}

pub struct SpeculativeVADCoordinator {
    config: SpeculativeVADCoordinatorConfig,
    vad_node: SileroVADNode,
    sessions: Arc<PLMutex<HashMap<String, CoordinatorState>>>,
}

impl SpeculativeVADCoordinator {
    pub fn with_config(config: SpeculativeVADCoordinatorConfig) -> Self {
        let vad_node = SileroVADNode::new(SileroVADConfig {
            model_path: None,
            threshold: config.vad_threshold,
            neg_threshold: None,
            sampling_rate: config.sample_rate,
            min_speech_duration_ms: config.min_speech_duration_ms,
            min_silence_duration_ms: config.min_silence_duration_ms,
            speech_pad_ms: config.speech_pad_ms,
        });
        Self {
            config,
            vad_node,
            sessions: Arc::new(PLMutex::new(HashMap::new())),
        }
    }

    pub fn new() -> Self {
        Self::with_config(SpeculativeVADCoordinatorConfig::default())
    }

    fn ensure_session_exists(&self, session_id: &str) {
        let mut sessions = self.sessions.lock();
        if !sessions.contains_key(session_id) {
            let samples_per_ms = self.config.sample_rate / 1000;
            let buffer_capacity = (self.config.lookback_ms * samples_per_ms) as usize;
            sessions.insert(
                session_id.to_string(),
                CoordinatorState::new(buffer_capacity),
            );
        }
    }

    fn current_timestamp_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| std::time::Duration::from_millis(0))
            .as_millis() as u64
    }
}

impl Default for SpeculativeVADCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AsyncStreamingNode for SpeculativeVADCoordinator {
    fn node_type(&self) -> &str {
        "SpeculativeVADCoordinator"
    }

    async fn process(&self, _data: RuntimeData) -> Result<RuntimeData, Error> {
        Err(Error::Execution(
            "SpeculativeVADCoordinator requires streaming mode - use process_streaming()".into(),
        ))
    }

    async fn process_streaming<F>(
        &self,
        data: RuntimeData,
        session_id: Option<String>,
        mut callback: F,
    ) -> Result<usize, Error>
    where
        F: FnMut(RuntimeData) -> Result<(), Error> + Send,
    {
        let session_id = session_id.unwrap_or_else(|| "default".to_string());

        let (samples, sample_rate, channels) = match &data {
            RuntimeData::Audio {
                samples,
                sample_rate,
                channels,
                ..
            } => (samples.clone(), *sample_rate, *channels),
            _ => {
                return Err(Error::Execution(
                    "SpeculativeVADCoordinator requires audio input".into(),
                ))
            }
        };

        let mut output_count = 0;

        // Step 1: Forward audio immediately (speculative)
        callback(RuntimeData::Audio {
            samples: samples.clone(),
            sample_rate,
            channels,
            stream_id: None,
            timestamp_us: None,
            arrival_ts_us: None,
            metadata: None,
        })?;
        output_count += 1;

        self.ensure_session_exists(&session_id);

        // Step 2: Buffer audio for potential cancellation
        {
            let mut sessions = self.sessions.lock();
            let state = sessions.get_mut(&session_id).unwrap();
            for &sample in samples.iter() {
                if state.audio_buffer.len() >= state.buffer_capacity {
                    state.audio_buffer.pop_front();
                }
                state.audio_buffer.push_back(sample);
            }
        }

        // Step 3: Run VAD inference (lock released)
        let vad_result: Option<Value> = {
            let vad_events: Arc<std::sync::Mutex<Vec<Value>>> =
                Arc::new(std::sync::Mutex::new(Vec::new()));
            let vad_events_cb = Arc::clone(&vad_events);
            let cb = move |out: RuntimeData| -> Result<(), Error> {
                if let RuntimeData::Json(j) = out {
                    if let Ok(mut e) = vad_events_cb.lock() {
                        e.push(j);
                    }
                }
                Ok(())
            };
            let _ = self
                .vad_node
                .process_streaming(data.clone(), Some(format!("{session_id}_vad")), cb)
                .await;
            vad_events.lock().ok().and_then(|e| e.first().cloned())
        };

        // Step 4: Process VAD result and track speech segments
        let mut pending_outputs: Vec<RuntimeData> = Vec::new();
        if let Some(vad_json) = vad_result {
            let has_speech = vad_json
                .get("has_speech")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let is_speech_start = vad_json
                .get("is_speech_start")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let is_speech_end = vad_json
                .get("is_speech_end")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let speech_probability = vad_json
                .get("speech_probability")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;

            {
                let mut sessions = self.sessions.lock();
                let state = sessions.get_mut(&session_id).unwrap();

                if is_speech_start {
                    state.speech_start_sample = Some(state.current_sample);
                    state.speech_triggered = true;
                    state.silence_samples = 0;
                }

                if !has_speech && state.speech_triggered {
                    state.silence_samples += samples.len();
                } else if has_speech {
                    state.silence_samples = 0;
                }

                if is_speech_end {
                    if let Some(start_sample) = state.speech_start_sample.take() {
                        let duration_samples = state.current_sample - start_sample;
                        let duration_ms = (duration_samples as f32 / self.config.sample_rate as f32
                            * 1000.0) as u32;

                        if duration_ms < self.config.min_speech_duration_ms {
                            let segment_id = format!("{}_{}", session_id, state.segment_counter);
                            state.segment_counter += 1;

                            pending_outputs.push(RuntimeData::ControlMessage {
                                message_type: ControlMessageType::CancelSpeculation {
                                    from_timestamp: start_sample as u64,
                                    to_timestamp: state.current_sample as u64,
                                },
                                segment_id: Some(segment_id.clone()),
                                timestamp_ms: Self::current_timestamp_ms(),
                                metadata: serde_json::json!({
                                    "reason": "speech_too_short",
                                    "duration_ms": duration_ms,
                                    "min_required_ms": self.config.min_speech_duration_ms,
                                    "vad_confidence": speech_probability,
                                }),
                            });

                            state.speculations_cancelled += 1;
                            tracing::info!(
                                session_id = %session_id,
                                segment_id = %segment_id,
                                duration_ms = duration_ms,
                                acceptance_rate = state.acceptance_rate() * 100.0,
                                "Speculation cancelled (false positive)"
                            );
                        } else {
                            state.speculations_accepted += 1;
                            state.audio_buffer.clear();
                            tracing::info!(
                                session_id = %session_id,
                                duration_ms = duration_ms,
                                acceptance_rate = state.acceptance_rate() * 100.0,
                                "Speculation accepted (confirmed speech)"
                            );
                        }
                    }
                    state.speech_triggered = false;
                    state.silence_samples = 0;
                }

                state.current_sample += samples.len();
            }

            pending_outputs.push(RuntimeData::Json(vad_json));
        } else {
            let mut sessions = self.sessions.lock();
            if let Some(state) = sessions.get_mut(&session_id) {
                state.current_sample += samples.len();
            }
        }

        for out in pending_outputs {
            callback(out)?;
            output_count += 1;
        }

        Ok(output_count)
    }

    async fn process_control_message(
        &self,
        message: RuntimeData,
        _session_id: Option<String>,
    ) -> Result<bool, Error> {
        match message {
            RuntimeData::ControlMessage { message_type, .. } => match message_type {
                ControlMessageType::CancelSpeculation { .. } => Ok(true),
                ControlMessageType::BatchHint { .. } => Ok(false),
                ControlMessageType::DeadlineWarning { .. } => Ok(false),
            },
            _ => Ok(false),
        }
    }
}

#[derive(Default)]
pub struct SpeculativeVADCoordinatorFactory;

impl FfiNodeFactory for SpeculativeVADCoordinatorFactory {
    fn node_type(&self) -> RString {
        RString::from("SpeculativeVADCoordinator")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        let cfg: SpeculativeVADCoordinatorConfig =
            serde_json::from_str(params.as_str()).unwrap_or_default();
        ROk(FfiNode_TO::from_value(
            StreamingNodeFfiAdapter::new(SpeculativeVADCoordinator::with_config(cfg)),
            TD_Opaque,
        ))
    }
}

// ---------------------------------------------------------------------------
// plugin registration
// ---------------------------------------------------------------------------

// Silence unused warning for AudioSamples (re-exported for completeness).
#[allow(dead_code)]
fn _audio_samples_used(_: AudioSamples) {}

// Emits the abi_stable root-module symbol the host's dlopen path
// looks up. Gated behind the `plugin-export` cargo feature
// (default-on) so consumers that link this crate as an rlib alongside
// other plugins can disable it to avoid duplicate-symbol collisions.
#[cfg(feature = "plugin-export")]
remotemedia_plugin_sdk::plugin_export!(SileroVADNodeFactory, SpeculativeVADCoordinatorFactory);
