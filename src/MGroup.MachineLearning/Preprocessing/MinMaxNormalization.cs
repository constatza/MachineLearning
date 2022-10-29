using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning.Preprocessing
{
	/// <summary>
	/// Normalize the data using the
	/// MinMax normalization so that
	/// their values lie in
	/// the [0,1] domain.
	/// </summary>
	public class MinMaxNormalization : INormalization
	{
		public double[] minValuePerDim { get; private set; }
		public double[] maxValuePerDim { get; private set; }
		public int dimension { get; private set; }
        public double[] ScalingRatio { get; private set; }

        public void Initialize(double[,] data, int dim)
		{
			dimension = dim;
			if (dimension == 0)
			{
				minValuePerDim = new double[data.GetLength(0)];
				maxValuePerDim = new double[data.GetLength(0)];

				for (int row = 0; row < data.GetLength(0); row++)
				{
					minValuePerDim[row] = double.MaxValue;
					maxValuePerDim[row] = double.MinValue;

					for (int col = 0; col < data.GetLength(1); col++)
					{
						if (data[row, col] < minValuePerDim[row])
						{
							minValuePerDim[row] = data[row, col];
						}

						if (data[row, col] > maxValuePerDim[row])
						{
							maxValuePerDim[row] = data[row, col];
						}
					}
				}
			}
			else if (dimension == 1)
			{
				minValuePerDim = new double[data.GetLength(1)];
				maxValuePerDim = new double[data.GetLength(1)];
				ScalingRatio = new double[data.GetLength(1)];

				for (int col = 0; col < data.GetLength(1); col++)
				{
					minValuePerDim[col] = double.MaxValue;
					maxValuePerDim[col] = double.MinValue;

					for (int row = 0; row < data.GetLength(0); row++)
					{
						if (data[row, col] < minValuePerDim[col])
						{
							minValuePerDim[col] = data[row, col];
						}

						if (data[row, col] > maxValuePerDim[col])
						{
							maxValuePerDim[col] = data[row, col];
						}
					}

					ScalingRatio[col] = maxValuePerDim[col] - minValuePerDim[col];
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
						scaledData[row, col] = (data[row, col] - minValuePerDim[row]) / (maxValuePerDim[row] - minValuePerDim[row]);
					}
				}
				return scaledData;
			}
			else if (dimension == 1)
			{
				for (int col = 0; col < data.GetLength(1); col++)
				{
					for (int row = 0; row < data.GetLength(0); row++)
					{
						scaledData[row, col] = (data[row, col] - minValuePerDim[col]) / (maxValuePerDim[col] - minValuePerDim[col]);
					}
				}
				return scaledData;
			}
			else
			{
				throw new ArgumentException("parameter 'dim' should be 0 or 1");
			}

		}

		public double[,] Denormalize(double[,] scaledData)
		{
			if (dimension == 0)
			{
				double[,] data = new double[scaledData.GetLength(0), scaledData.GetLength(1)];
				for (int row = 0; row < scaledData.GetLength(0); row++)
				{
					for (int col = 0; col < scaledData.GetLength(1); col++)
					{
						data[row, col] = scaledData[row, col] * (maxValuePerDim[row] - minValuePerDim[row]) + minValuePerDim[row];
					}
				}
				return data;
			}
			else if (dimension == 1)
			{
				double[,] data = new double[scaledData.GetLength(0), scaledData.GetLength(1)];
				for (int col = 0; col < scaledData.GetLength(1); col++)
				{
					for (int row = 0; row < scaledData.GetLength(0); row++)
					{
						data[row, col] = scaledData[row, col] * (maxValuePerDim[col] - minValuePerDim[col]) + minValuePerDim[col];
					}
				}
				return data;
			}
			else
			{
				throw new ArgumentException("parameter 'dim' should be 0 or 1");
			}
		}
	}
}
