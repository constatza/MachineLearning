using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning
{
	public interface INormalization
	{
		// dim is actually a flag (0: leave as is, 1: tranpose the input)
		public void Initialize(double[,] X, int dim);

		public double[,] Normalize(double[,] X);

	}
}
