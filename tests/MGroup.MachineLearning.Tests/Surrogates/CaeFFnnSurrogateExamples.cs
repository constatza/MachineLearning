namespace MGroup.MachineLearning.Tests
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
	using MGroup.MachineLearning.Tests.Surrogates;

	public static class CaeFffnnExamples
	{
		public static void InvestigateFfnnStatistics()
		{
			string outputPath = "ffnn_errors_NET.txt";
			(double[,] solutions, double[,] parameters, double[,] latentSpace) = CaeFffnnSurrogateTests.ReadData();
			var ffnn = new FFNN(8);

			var evaluator = new SurrogateModelEvaluator(ffnn, 1000, outputPath);
			evaluator.RunExperiments(parameters, latentSpace);
			(int sampleSize, double mean, double stdev) = evaluator.PerformStatisticAnalysisOnErrors(evaluator.OutputFilePaths[0]);
			Debug.WriteLine($"Errors: sample size = {sampleSize}, mean = {mean}, stdev = {stdev}");
		}

		public static void InvestigateFullSurrogateStatistics()
		{
			string outputPath = "surrogate_errors_NET.txt";
			(double[,] solutions, double[,] parameters, double[,] latentSpace) = CaeFffnnSurrogateTests.ReadData();
			var surrogateBuilder = new CaeFffnSurrogate.Builder();
			surrogateBuilder.CaeNumEpochs = 40;
			CaeFffnSurrogate surrogate = surrogateBuilder.BuildSurrogate();

			var evaluator = new SurrogateModelEvaluator(surrogate, 1000, outputPath);
			evaluator.RunExperiments(parameters, solutions);

			(int sampleSize, double mean, double stdev) = evaluator.PerformStatisticAnalysisOnErrors(evaluator.OutputFilePaths[0]);
			Debug.WriteLine($"CAE errors: sample size = {sampleSize}, mean = {mean}, stdev = {stdev}");
			(sampleSize, mean, stdev) = evaluator.PerformStatisticAnalysisOnErrors(evaluator.OutputFilePaths[1]);
			Debug.WriteLine($"Full surrogate errors: sample size = {sampleSize}, mean = {mean}, stdev = {stdev}");
		}
	}
}
