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

        public void Initialize(double[,] data, int dim)
        {
			meanValuePerDim = new double[data.GetLength(0)];
			stdValuePerDim = new double[data.GetLength(0)];
			for (int row = 0; row < data.GetLength(0); row++)
			{
				for (int col = 0; col < data.GetLength(1); col++)
				{
					meanValuePerDim[row] += data[row, col];
				}
				meanValuePerDim[row] = meanValuePerDim[row] / data.GetLength(1);
			}

			for (int row = 0; row < data.GetLength(0); row++)
			{
				for (int col = 0; col < data.GetLength(1); col++)
				{
					stdValuePerDim[row] += Math.Pow(data[row, col] - meanValuePerDim[row], 2);
				}
				stdValuePerDim[row] = Math.Sqrt(stdValuePerDim[row] / (data.GetLength(1) - 1));
			}
		}

        public double[,] Normalize(double[,] data)
		{
			double[,] scaledData = new double[data.GetLength(0), data.GetLength(1)];

			for (int row = 0; row < data.GetLength(0); row++)
			{
				for (int col = 0; col < data.GetLength(1); col++)
				{
					scaledData[row, col] = (data[row, col] - meanValuePerDim[row]) / stdValuePerDim[row];
				}
			}

			return (scaledData);
		}
    }
}
