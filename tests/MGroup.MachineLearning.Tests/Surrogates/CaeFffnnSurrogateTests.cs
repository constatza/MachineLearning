namespace MGroup.MachineLearning.Tests.Surrogates
{
	using System;
	using System.Collections.Generic;
	using System.Diagnostics;
	using System.Linq;
	using System.Text;
	using System.Threading.Tasks;

	using MGroup.MachineLearning.Tests.Utilities;
	using MGroup.MachineLearning.TensorFlow;
	using MGroup.MachineLearning.TensorFlow.KerasLayers;
	using MGroup.MachineLearning.Utilities;

	using NumSharp;

	using Tensorflow;

	using Xunit;
	using System.IO;

	[Collection("Run sequentially")]
	public static class CaeFffnnSurrogateTests
	{
		[Fact]
		public static void TestBiot()
		{
			// Do NOT execute on Azure DevOps
			if (Environment.GetEnvironmentVariable("SYSTEM_DEFINITIONID") != null)
			{
				return;
			}

			(var solutions, var parameters, var latentSpace) = ReadData();
			var numSamples = solutions.GetLength(0);
			Assert.Equal(parameters.GetLength(0), numSamples);
			var solutionSpaceDim = solutions.GetLength(1);
			var parametricSpaceDim = parameters.GetLength(1);

			var seed = 1234;
			//tf.set_random_seed(seed);
			var surrogateBuilder = new CaeFffnSurrogate.Builder();
			surrogateBuilder.TensorFlowSeed = seed;
			var surrogate = surrogateBuilder.BuildSurrogate();
			var errors = surrogate.TrainAndEvaluate(parameters, solutions, surrogateBuilder.Splitter);

			var input = GetRow(0, parameters);
			var prediction = surrogate.Predict(input);
			var surrogateError = errors["Surrogate error"];
			Assert.InRange(surrogateError, 0, 0.3);
			var caeError = errors["CAE error"];
			Assert.InRange(caeError, 0, 0.08);
		}

		internal static (double[,] solutions, double[,] parameters, double[,] latentSpace) ReadData()
		{
			var folder = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.FullName
				+ "\\MGroup.MachineLearning.Tests\\SavedFiles\\CaeFfnnSurrogate\\";
			var solutionsPath = folder + "solutions.npy";
			var parametersPath = folder + "parameters.npy";
			var latentSpacePath = folder + "latentSpace.npy";

			var solutionVectors = np.Load<double[,,]>(solutionsPath); // [numSamples, 1, numDofs]
			var parameters = np.Load<double[,]>(parametersPath); // [numSamples, numParameters]
			var latentSpace = np.Load<double[,]>(latentSpacePath); // [numSamples, latentSpaceSize]

			return (solutionVectors.RemoveEmptyDimension(1), parameters, latentSpace);
		}

		private static double[] GetRow(int rowIdx, double[,] matrix)
		{
			var n = matrix.GetLength(1);
			var result = new double[n];
			for (var j = 0; j < n; j++)
			{
				result[j] = matrix[rowIdx, j];
			}
			return result;
		}
	}
}
