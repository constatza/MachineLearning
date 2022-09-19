using System.Linq;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.NumPy;
using Xunit;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples
{
    /// <summary>
    /// The Keras functional API is a way to create models that are more flexible than the tf.keras.Sequential API. 
    /// The functional API can handle models with non-linear topology, shared layers, and even multiple inputs or outputs.
    /// https://keras.io/guides/functional_api/
    /// </summary>
    public class MnistFnnKerasFunctional
    {
        Model model;
        NDArray x_train, y_train, x_test, y_test;

        [Fact]
        public bool Run()
        {
            tf.enable_eager_execution();

            PrepareData();
            BuildModel();
            Train();
            Predict();

            return true;
        }

        public void PrepareData()
        {
            (x_train, y_train, x_test, y_test) = keras.datasets.mnist.load_data();
            x_train = x_train.reshape((60000, 784)) / 255f;
            x_test = x_test.reshape((10000, 784)) / 255f;
        }

        public void BuildModel()
        {
            // input layer
            var inputs = keras.Input(shape: 784);

            var layers = new LayersApi();

            // 1st dense layer
            var outputs = layers.Dense(64, activation: keras.activations.Relu).Apply(inputs);

            // 2nd dense layer
            outputs = layers.Dense(64, activation: keras.activations.Relu).Apply(outputs);

            // output layer
            outputs = layers.Dense(10).Apply(outputs);

            // build keras model
            model = keras.Model(inputs, outputs, name: "mnist_model");
            // show model summary
            model.summary();

            // compile keras model into tensorflow's static graph
            model.compile(loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                optimizer: keras.optimizers.RMSprop(),
                metrics: new[] { "accuracy" });
        }

        public void Train()
        {
            // train model by feeding data and labels.
            model.fit(x_train, y_train, batch_size: 64, epochs: 2);

            // evluate the model
            model.evaluate(x_test, y_test, batch_size: 128);
            
            // save and serialize model
            model.save("mnist_model");

            // recreate the exact same model purely from the file:
            // model = keras.models.load_model("path_to_my_model");
        }

        public void Predict()
        {
            var pred1 = model.predict(x_test);
            var pred2 = model.Apply(x_test, training: false);
        }
    }
}
