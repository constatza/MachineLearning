using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning
{
	public interface INormalization
	{

		public void Initialize(double[,] X, int dim);

		public double[,] Normalize(double[,] X);

	}
}
