using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning
{
	public interface INormalization
	{
		double[,] Normalize(double[,] X,int dim);
	}
}
