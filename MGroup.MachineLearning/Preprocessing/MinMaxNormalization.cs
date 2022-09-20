using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning
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

		public double[,] Normalize(double[,] data, int dim)
		{
			if (dim == 0)
			{
				double[,] scaledData= new double[data.GetLength(0), data.GetLength(1)];
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

				for (int row = 0; row < data.GetLength(0); row++)
				{
					for (int col = 0; col < data.GetLength(1); col++)
					{
						scaledData[row, col] = (data[row, col] - minValuePerDim[row]) / (maxValuePerDim[row] - minValuePerDim[row]);
					}
				}
				return scaledData;
			}
            else if(dim == 1)
            {
				double[,] scaledData = new double[data.GetLength(0), data.GetLength(1)];
				minValuePerDim = new double[data.GetLength(1)];
				maxValuePerDim = new double[data.GetLength(1)];		
				
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
				}

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

		public double[,] Denormalize(double[,] data, int dim)
		{
			if (dim == 0)
			{
				double[,] scaledData = new double[data.GetLength(0), data.GetLength(1)];
				for (int row = 0; row < data.GetLength(0); row++)
				{
					for (int col = 0; col < data.GetLength(1); col++)
					{
						scaledData[row, col] = data[row, col] * (maxValuePerDim[row] - minValuePerDim[row]) + minValuePerDim[row];
					}
				}
				return scaledData;
			}
			else if (dim == 1)
			{
				double[,] scaledData = new double[data.GetLength(0), data.GetLength(1)];
				for (int col = 0; col < data.GetLength(1); col++)
				{
					for (int row = 0; col < data.GetLength(0); row++)
					{
						scaledData[row, col] = data[row, col] * (maxValuePerDim[col] - minValuePerDim[col]) + minValuePerDim[col];
					}
				}
				return scaledData;
			}
			else
			{
				throw new ArgumentException("parameter 'dim' should be 0 or 1");
			}
		}
	}
}
