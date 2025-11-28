#include <Arduino.h>
#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---------------------- Globals ----------------------
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 30 * 1024; // adjust if model too big
uint8_t tensor_arena[kTensorArenaSize];

// ---------------------- Scaling Parameters (from Python StandardScaler) ----------------------
float mean[5] = {24.5, 52.3, 405.0, 0.012, 1010.5};
float std_dev[5]  = {2.1, 5.0, 50.0, 0.005, 10.0};

// ---------------------- Threshold ----------------------
float fire_threshold = 0.45; // tune as needed
}

// ---------------------- Setup ----------------------
void setup() {
Serial.begin(115200);
while (!Serial) {}

// Setup TFLite Micro error reporter
static tflite::MicroErrorReporter micro_error_reporter;
error_reporter = &micro_error_reporter;

// Load the TFLite model
model = tflite::GetModel(g_model);
if (model->version() != TFLITE_SCHEMA_VERSION) {
error_reporter->Report("Model schema %d not equal to %d",
model->version(), TFLITE_SCHEMA_VERSION);
return;
}

// Setup operations resolver
static tflite::AllOpsResolver resolver;

// Setup interpreter
static tflite::MicroInterpreter static_interpreter(
model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
interpreter = &static_interpreter;

// Allocate memory for tensors
if (interpreter->AllocateTensors() != kTfLiteOk) {
error_reporter->Report("AllocateTensors failed");
return;
}

input = interpreter->input(0);
output = interpreter->output(0);

Serial.println("TFLite Micro initialized!");
Serial.println("\n*** FIRE DETECTION SYSTEM STARTED ***\n");
}

// ---------------------- Function to run inference ----------------------
void runInference(float temperature, float humidity, float co2, float hydrogen, float pressure, const char* scenario) {
float raw_inputs[5] = {temperature, humidity, co2, hydrogen, pressure};
float scaled_inputs[5];

// ---------------------- PRINT SENSOR DATA ----------------------
Serial.println("\n========================================");
Serial.print("TEST SCENARIO: ");
Serial.println(scenario);
Serial.println("========================================");
Serial.print("Temperature: ");
Serial.print(temperature);
Serial.println(" °C");
Serial.print("Humidity: ");
Serial.print(humidity);
Serial.println(" %");
Serial.print("CO2: ");
Serial.print(co2);
Serial.println(" ppm");
Serial.print("Hydrogen: ");
Serial.print(hydrogen);
Serial.println(" %");
Serial.print("Pressure: ");
Serial.print(pressure);
Serial.println(" hPa");

// ---------------------- APPLY STANDARD SCALER ----------------------
for (int i = 0; i < 5; i++) {
scaled_inputs[i] = (raw_inputs[i] - mean[i]) / std_dev[i];
}

// ---------------------- PRINT SCALED INPUTS ----------------------
Serial.println("\n--- SCALED INPUTS ---");
Serial.print("Temp: ");
Serial.print(scaled_inputs[0], 4);
Serial.print(" | Humidity: ");
Serial.print(scaled_inputs[1], 4);
Serial.print(" | CO2: ");
Serial.print(scaled_inputs[2], 4);
Serial.print(" | H2: ");
Serial.print(scaled_inputs[3], 4);
Serial.print(" | Pressure: ");
Serial.println(scaled_inputs[4], 4);

// ---------------------- QUANTIZE INPUTS ----------------------
for (int i = 0; i < 5; i++) {
int8_t q = (int8_t)(scaled_inputs[i] / input->params.scale + input->params.zero_point);
input->data.int8[i] = q;
}

// ---------------------- RUN INFERENCE ----------------------
if (interpreter->Invoke() != kTfLiteOk) {
Serial.println("Invoke failed!");
return;
}

// ---------------------- DEQUANTIZE OUTPUT ----------------------
int8_t y_q = output->data.int8[0];
float fire_probability = (y_q - output->params.zero_point) * output->params.scale;

// Clamp to [0, 1]
if (fire_probability < 0.0f) fire_probability = 0.0f;
if (fire_probability > 1.0f) fire_probability = 1.0f;

// ---------------------- CALCULATE ADDITIONAL METRICS ----------------------
float no_fire_probability = 1.0f - fire_probability;
float confidence = (fire_probability > no_fire_probability) ? fire_probability : no_fire_probability;

// ---------------------- PRINT PREDICTIONS ----------------------
Serial.println("\n--- PREDICTIONS ---");
Serial.print("Fire Probability: ");
Serial.print(fire_probability * 100, 2);
Serial.print("% | No Fire: ");
Serial.print(no_fire_probability * 100, 2);
Serial.print("% | Confidence: ");
Serial.print(confidence * 100, 2);
Serial.println("%");

// ---------------------- DETECTION RESULT ----------------------
Serial.print("RESULT: ");
if (fire_probability > fire_threshold) {
Serial.println("🔥 FIRE DETECTED!");
} else {
Serial.println("✓ NO FIRE");
}
}

// ---------------------- Loop ----------------------
void loop() {
// Test Scenario 1: Normal conditions (no fire)
runInference(20.0, 45.0, 380.0, 0.008, 1013.0, "Normal Conditions");
delay(3000);

// Test Scenario 2: High temperature (potential fire)
runInference(45.0, 30.0, 450.0, 0.05, 1010.0, "High Temperature");
delay(3000);

// Test Scenario 3: High CO2 + High Hydrogen (fire indicators)
runInference(35.0, 60.0, 800.0, 0.15, 1008.0, "High CO2 + Hydrogen");
delay(3000);

// Test Scenario 4: Extreme conditions (strong fire signal)
runInference(80.0, 15.0, 1200.0, 0.30, 990.0, "Extreme Fire Conditions");
delay(3000);

// Test Scenario 5: Cold with high humidity (no fire)
runInference(5.0, 80.0, 350.0, 0.002, 1020.0, "Cold + High Humidity");
delay(3000);

// Test Scenario 6: Moderate fire indicators
runInference(32.0, 55.0, 550.0, 0.08, 1012.0, "Moderate Fire Indicators");
delay(3000);
}