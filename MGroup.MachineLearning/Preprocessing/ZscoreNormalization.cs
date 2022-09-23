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
		public int dimension { get; private set; }

		public void Initialize(double[,] data, int dim)
        {
			if (dimension == 0)
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
			else if(dimension == 1)
            {
				meanValuePerDim = new double[data.GetLength(1)];
				stdValuePerDim = new double[data.GetLength(1)];
				for (int col = 0; col < data.GetLength(0); col++)
				{
					for (int row = 0; row < data.GetLength(1); row++)
					{
						meanValuePerDim[col] += data[row, col];
					}
					meanValuePerDim[col] = meanValuePerDim[col] / data.GetLength(0);
				}

				for (int col = 0; col < data.GetLength(0); col++)
				{
					for (int row = 0; row < data.GetLength(1); row++)
					{
						stdValuePerDim[col] += Math.Pow(data[row, col] - meanValuePerDim[col], 2);
					}
					stdValuePerDim[col] = Math.Sqrt(stdValuePerDim[col] / (data.GetLength(0) - 1));
				}
			}
		}

        public double[,] Normalize(double[,] data)
		{
			double[,] scaledData = new double[data.GetLength(0), data.GetLength(1)];

			if (dimension == 0)
			{
				for (int row = 0; row < data.GetLength(0); row++)
				{
					for (int col = 0; col < data.GetLength(1); col++)
					{
						scaledData[row, col] = (data[row, col] - meanValuePerDim[row]) / stdValuePerDim[row];
					}
				}
			}
			else if (dimension == 1)
			{
				for (int col = 0; col < data.GetLength(0); col++)
				{
					for (int row = 0; row < data.GetLength(1); row++)
					{
						scaledData[row, col] = (data[row, col] - meanValuePerDim[col]) / stdValuePerDim[col];
					}
				}
			}

			return (scaledData);
		}
    }
}
