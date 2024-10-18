import tensorflow as tf

# Load the frozen graph
model_path = "frozen_inference_graph.pb"  # Ensure this path is correct

# Load the graph
with tf.io.gfile.GFile(model_path, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Import the graph into a new TensorFlow graph
tf.compat.v1.import_graph_def(graph_def, name='')

# Create a concrete function from the graph
input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('normalized_input_image_tensor:0')
output_boxes = tf.compat.v1.get_default_graph().get_tensor_by_name('raw_outputs/box_encodings:0')
output_classes = tf.compat.v1.get_default_graph().get_tensor_by_name('raw_outputs/class_predictions:0')

# Use a TensorFlow function for conversion
@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32)])
def serving_fn(input_tensor):
    return tf.concat([output_boxes, output_classes], axis=-1)

# Convert the function to TFLite model
converter = tf.lite.TFLiteConverter.from_concrete_functions([serving_fn.get_concrete_function()])

# Optimize the model (optional)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model to a file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TFLite and saved as model.tflite.")
