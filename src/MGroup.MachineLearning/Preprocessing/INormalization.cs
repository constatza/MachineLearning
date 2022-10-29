using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning.Preprocessing
{
	public interface INormalization
	{
        // dim is actually a flag (0: leave as is, 1: tranpose the input)
        void Initialize(double[,] X, int dim);

        double[,] Normalize(double[,] X);

        double[,] Denormalize(double[,] X);

        double[] ScalingRatio { get; }

    }
}
