using System;
using System.IO;

using MGroup.Constitutive.Structural.MachineLearning;
using MGroup.MachineLearning.Preprocessing;
using MGroup.MachineLearning.TensorFlow.NeuralNetworks;
using static Tensorflow.KerasApi;

using Xunit;
using MGroup.Constitutive.Structural.Continuum;
using MGroup.LinearAlgebra.Matrices;
using System.Reflection;

namespace MGroup.MachineLearning.Tests
{
    public static class NeuralNetworkMaterial3DTest
	{
		[Fact]
		public static void RunTest()
		{
			// these files are used to generate an already trained FeedForwardNeuralNetwork which was created with strain-stress pairs from an ElasticMaterial3D(youngModulus:20, poissonRatio:0.2)
			var netPathName = "SavedFiles\\network_architecture.txt";
			var weightsPathName = "SavedFiles\\trained_weights";
			var normalizationXPathName = "SavedFiles\\normalizationX";
			var normalizationYPathName = "SavedFiles\\normalizationY";

			var neuralNetwork = new FeedForwardNeuralNetwork();
			neuralNetwork.LoadNetwork(netPathName, weightsPathName, normalizationXPathName, normalizationYPathName);

			var neuralNetworkMaterial = new NeuralNetworkMaterial3D(neuralNetwork, new double[0]);
			var elasticMaterial = new ElasticMaterial3D(20, 0.2);

			CheckNeuralNetworkMaterialAccuracy(neuralNetworkMaterial, elasticMaterial);
		}

		private static void CheckNeuralNetworkMaterialAccuracy(NeuralNetworkMaterial3D neuralNetworkMaterial, ElasticMaterial3D elasticMaterial)
		{
			var increments = 5;
			var cases = 6;
			var stressesNeuralNetwork = new double[cases * increments, 6];
			var constitutiveNeuralNetwork = new Matrix[cases * increments];
			var stressesElastic = new double[cases * increments, 6];
			var constitutiveElastic = new Matrix[cases * increments];
			var strainCases = new double[6][] { new double[6]{ 0.001, 0.00, 0.00, 0.00, 0.00, 0.00 }, new double[6] { 0.00, 0.001, 0.00, 0.00, 0.00, 0.00 }, new double[6] { 0.00, 0.00, 0.001, 0.00, 0.00, 0.00 },
												new double[6]{ 0.00, 0.00, 0.00, 0.001, 0.00, 0.00 }, new double[6]{ 0.00, 0.00, 0.00, 0.00, 0.001, 0.00 }, new double[6]{ 0.00, 0.00, 0.00, 0.00, 0.00, 0.001 }};
			for (int k = 0; k < strainCases.GetLength(0); k++)
			{
				for (int i = 0; i < increments; i++)
				{

					elasticMaterial.UpdateConstitutiveMatrixAndEvaluateResponse(strainCases[k]);
					neuralNetworkMaterial.UpdateConstitutiveMatrixAndEvaluateResponse(strainCases[k]);
					for (int j = 0; j < 6; j++)
					{
						stressesNeuralNetwork[k * increments + i, j] = neuralNetworkMaterial.Stresses[j];
						stressesElastic[k * increments + i, j] = elasticMaterial.Stresses[j];
					}
					constitutiveNeuralNetwork[k * increments + i] = (Matrix)neuralNetworkMaterial.ConstitutiveMatrix;
					constitutiveElastic[k * increments + i] = (Matrix)elasticMaterial.ConstitutiveMatrix;
					neuralNetworkMaterial.CreateState();
					elasticMaterial.CreateState();
				}
				neuralNetworkMaterial.ClearState();
				elasticMaterial.ClearState();
			}

			var stressDeviation = 0d;
			var constitutiveDeviation = 0d;
			for (int k = 0; k < strainCases.GetLength(0); k++)
			{
				for (int i = 0; i < increments; i++)
				{
					for (int j1 = 0; j1 < 6; j1++)
					{
						stressDeviation += Math.Pow((stressesNeuralNetwork[k * increments + i, j1] - stressesElastic[k * increments + i, j1]), 2);
						for (int j2 = 0; j2 < 6; j2++)
						{
							constitutiveDeviation += Math.Pow((constitutiveNeuralNetwork[k * increments + i][j1, j2] - constitutiveElastic[k * increments + i][j1, j2]), 2);
						}
					}
				}
			}

			stressDeviation = stressDeviation / (increments * strainCases.GetLength(0));
			constitutiveDeviation = constitutiveDeviation / (increments * strainCases.GetLength(0));

			Assert.True(stressDeviation < 1e-6 && constitutiveDeviation < 2e-1);
		}
	}
}
