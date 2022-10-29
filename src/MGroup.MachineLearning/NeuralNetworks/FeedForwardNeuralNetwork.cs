using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;
using Tensorflow.NumPy;
using Tensorflow.Keras.Layers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Losses;
using System.IO;
using System.Xml.Serialization;

namespace MGroup.MachineLearning.NeuralNetworks
{
    public class FeedForwardNeuralNetwork : INeuralNetwork
    {
        public INormalization Normalization { get; set; }

        //Optimization Parameters
        public OptimizerV2 Optimizer { get; set; }
        public ILossFunc LossFunction { get; set; }
        public double LearningRate { get; set; }
        public int BatchSize { get; set; }
        public int Epochs { get; set; }
        public int DisplayStep { get; set; }
        public Layer[] Layer { get; private set; }

        //Neural Network Architecture
        public int NumHiddenLayers { get; set; }
        public int[] NumNeuronsPerLayer { get; set; }
        public string[] ActivationFunctionPerLayer { get; set; }

        public int? seed;

        Model model;
        NDArray trainX;
        NDArray testX;
        NDArray trainY;
        NDArray testY;

        public FeedForwardNeuralNetwork()
        {
            Optimizer = keras.optimizers.Adam();
            LearningRate = 0.001;
            BatchSize = -1;
            Epochs = 1;
            DisplayStep = 100;
            NumHiddenLayers = 1;
            NumNeuronsPerLayer = new int[] { 1 };
            ActivationFunctionPerLayer = new string[] { "linear" };
            Layer = new Layer[1] { new Dense(new DenseArgs() { Units = NumNeuronsPerLayer[0], Activation = KerasApi.keras.activations.Linear, DType = TF_DataType.TF_DOUBLE }) };
            if (seed != null)
            {
                tf.set_random_seed((int)seed);
            }
        }

        public void Train(double[,] trainX, double[,] trainY, double[,] testX = null, double[,] testY = null)
        {
            tf.enable_eager_execution();

            if (Normalization != null)
            {
                if (testX != null)
                {
                    var trainAndTestX = new double[trainX.GetLength(0) + testX.GetLength(0), trainX.GetLength(1)];
                    trainX.CopyTo(trainAndTestX, 0);
                    testX.CopyTo(trainAndTestX, trainX.Length);
                    Normalization.Initialize(trainAndTestX, dim: 1);
                    //
                    trainX = Normalization.Normalize(trainX);
                    testX = Normalization.Normalize(testX);
                }
                else
                {
                    Normalization.Initialize(trainX, dim: 1);
                    trainX = Normalization.Normalize(trainX);
                }
            }

            this.trainX = np.array(trainX);
            this.trainY = np.array(trainY);

            if (testX != null && testY != null)
            {
                this.testX = np.array(testX, TF_DataType.TF_DOUBLE);
                this.testY = np.array(testY, TF_DataType.TF_DOUBLE);
            }

            keras.backend.clear_session();
            keras.backend.set_floatx(TF_DataType.TF_DOUBLE);

            var inputs = keras.Input(shape: trainX.GetLength(1), dtype: TF_DataType.TF_DOUBLE);
            var outputs = new Dense(new DenseArgs() { Units = NumNeuronsPerLayer[0], Activation = GetActivationByName(ActivationFunctionPerLayer[0]), DType = TF_DataType.TF_DOUBLE })
                .Apply(inputs);

            for (int i = 1; i < NumHiddenLayers; i++)
            {
                outputs = new Dense(new DenseArgs() { Units = NumNeuronsPerLayer[i], Activation = GetActivationByName(ActivationFunctionPerLayer[i]), DType = TF_DataType.TF_DOUBLE })
                    .Apply(outputs);
            }

            outputs = new Dense(new DenseArgs() { Units = trainY.GetLength(1), Activation = GetActivationByName("linear"), DType = TF_DataType.TF_DOUBLE })
                .Apply(outputs);

            // build keras model
            model = new Keras.Model(inputs, outputs, "current_model");
            // show model summary
            model.summary();

            // compile keras model into tensorflow's static graph
            model.compile(loss: LossFunction,
                optimizer: Optimizer,
                metrics: new[] { "accuracy" });

            // train the model
            model.fit(this.trainX, this.trainY, batch_size: BatchSize, epochs: Epochs);

            // evaluate the model
            if (testX != null && testY != null)
            {
                model.evaluate(this.testX, this.testY, batch_size: BatchSize);
            }
        }

        public double[,] Predict(double[,] data)
        {
            if (Normalization != null)
            {
                (data) = Normalization.Normalize(data);
            }
            var npData = np.array(data, TF_DataType.TF_DOUBLE);
            // predict output of new data
            var resultSqueezed = tf.squeeze(model.Apply(npData, training: false)).ToArray<double>();
            var result = new double[data.GetLength(0), trainY.GetShape().as_int_list()[1]];
            for (int i = 0; i < result.GetLength(0); i++)
            {
                for (int j = 0; j < result.GetLength(1); j++)
                {
                    result[i, j] = resultSqueezed[trainY.GetShape().as_int_list()[1] * i + j];
                }
            }
            return result;
        }

        public double[,] Gradient(double[,] data)
        {
            if (Normalization != null)
            {
                (data) = Normalization.Normalize(data);
            }
            var npData = np.array(data, TF_DataType.TF_DOUBLE);
            npData = (NDArray)tf.constant(npData);
            using var tape = tf.GradientTape(persistent: true);
            {
                tape.watch(npData);
                var pred = model.Apply(npData, training: false);
                var numRowsGrad = tf.size(pred).ToArray<int>()[0];
                var numColsGrad = tf.size(npData).ToArray<int>()[0];
                var slicedPred = new Tensor();
                var gradient = new double[numRowsGrad, numColsGrad];
                for (int i = 0; i < numRowsGrad; i++)
                {
                    slicedPred = tf.slice<int, int>(pred, new int[] { 0, i }, new int[] { 1, 1 });
                    var slicedGrad = tape.gradient(slicedPred, npData).ToArray<double>();
                    for (int j = 0; j < numColsGrad; j++)
                    {
                        gradient[i, j] = slicedGrad[j];
                    }
                }
                return gradient;
            }
        }

        public void SaveNetwork(string netPath, string weightsPath)
        {
            using (var writer = new StreamWriter(netPath))
            {
                writer.WriteLine($"{trainX.GetShape().as_int_list()[1]}");
                for (int i = 0; i < NumHiddenLayers; i++)
                {
                    writer.WriteLine($"{NumNeuronsPerLayer[i]}");
                    writer.WriteLine($"{ActivationFunctionPerLayer[i]}");
                }
                writer.WriteLine($"{trainY.GetShape().as_int_list()[1]}");
            }
            model.save_weights(weightsPath);

            using (var writer = new StreamWriter(netPath + ".2.xml"))
            {
                XmlSerializer serializer = new XmlSerializer(typeof(List<ILayer>));
                serializer.Serialize(writer, model.Layers);
            }
        }

        public void LoadNetwork(string netPath, string weightsPath)
        {
            keras.backend.clear_session();

            var layers = new LayersApi();

            var lines = File.ReadAllLines(netPath);

            var inputs = keras.Input(shape: Convert.ToInt32(lines.First()));

            var outputs = layers.Dense(Convert.ToInt32(lines[1]), lines[2]).Apply(inputs);

            for (int i = 1; i < NumHiddenLayers; i++)
            {
                outputs = layers.Dense(Convert.ToInt32(lines[2 * i + 1]), lines[2 * i + 2]).Apply(outputs);
            }

            outputs = layers.Dense(Convert.ToInt32(lines.Last())).Apply(outputs);

            // build keras model
            model = keras.Model(inputs, outputs, name: "current_model");

            model.load_weights(weightsPath);
        }

        float[,] DoubleToFloat_DEPRECATED(double[,] matrix)
        {
            float[,] floatMatrix = new float[matrix.GetLength(0), matrix.GetLength(1)];
            for (int i = 0; matrix.GetLength(0) > i; i++)
            {
                for (int j = 0; matrix.GetLength(1) > j; j++)
                {
                    floatMatrix[i, j] = (float)matrix[i, j];
                }
            }
            return floatMatrix;
        }

        double[,] FloatToDouble(float[,] matrix)
        {
            double[,] doubleMatrix = new double[matrix.GetLength(0), matrix.GetLength(1)];
            for (int i = 0; matrix.GetLength(0) > i; i++)
            {
                for (int j = 0; matrix.GetLength(1) > j; j++)
                {
                    doubleMatrix[i, j] = (double)matrix[i, j];
                }
            }
            return doubleMatrix;
        }

        private Activation GetActivationByName(string name)
        {
            return name switch
            {
                "linear" => KerasApi.keras.activations.Linear,
                "relu" => KerasApi.keras.activations.Relu,
                "sigmoid" => KerasApi.keras.activations.Sigmoid,
                "tanh" => KerasApi.keras.activations.Tanh,
                "softmax" => KerasApi.keras.activations.Softmax,
                _ => throw new Exception("Activation " + name + " not found"),
            };
        }
    }
}
