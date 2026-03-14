

# Whether the operation succeeded
bool success

# Status message
string message

# Recorded audio samples (PCM float32, mono)
# Only populated when start=false (stop)
float32[] audio_data

# Sample rate in Hz (e.g., 16000)
int32 sample_rate

# Duration of recording in seconds
float64 duration
