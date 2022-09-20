using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning
{
	/// <summary>
	/// Normalize the data using the
	/// Z-score normalization so that
	/// their values have zero mean and unit variance.
	/// </summary>
	public class ZscoreNormalization : INormalization
	{
		public double[] meanValuePerDim { get; private set; }
		public double[] stdValuePerDim { get; private set; }

		public double[,] Normalize(double[,] rawData, int dim)
		{
			double[,] scaledData = new double[rawData.GetLength(0), rawData.GetLength(1)];
			double[] meanValuePerDim = new double[rawData.GetLength(0)];
			double[] stdValuePerDim = new double[rawData.GetLength(0)];
			for (int row = 0; row < rawData.GetLength(0); row++)
			{
				for (int col = 0; col < rawData.GetLength(1); col++)
				{
					meanValuePerDim[row] += rawData[row, col];
				}
				meanValuePerDim[row] = meanValuePerDim[row] / rawData.GetLength(1);
			}

			for (int row = 0; row < rawData.GetLength(0); row++)
			{
				for (int col = 0; col < rawData.GetLength(1); col++)
				{
					stdValuePerDim[row] += Math.Pow(rawData[row, col]- meanValuePerDim[row],2);
				}
				stdValuePerDim[row] = Math.Sqrt(stdValuePerDim[row] / (rawData.GetLength(1)-1));
			}

			for (int row = 0; row < rawData.GetLength(0); row++)
			{
				for (int col = 0; col < rawData.GetLength(1); col++)
				{
					scaledData[row, col] = (rawData[row, col] - meanValuePerDim[row]) / stdValuePerDim[row];
				}
			}

			return (scaledData);
		}
	}
}
