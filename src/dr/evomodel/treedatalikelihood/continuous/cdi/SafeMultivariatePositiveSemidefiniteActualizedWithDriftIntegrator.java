package dr.evomodel.treedatalikelihood.continuous.cdi;

import dr.math.matrixAlgebra.missingData.MissingOps;
import org.ejml.interfaces.decomposition.EigenDecomposition;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.EigenOps;

import static dr.math.matrixAlgebra.missingData.MissingOps.anyDiagonalInfinities;
import static dr.math.matrixAlgebra.missingData.MissingOps.unwrap;
import static dr.math.matrixAlgebra.missingData.MissingOps.weightedInnerProductOfDifferences;
import static org.ejml.data.DenseMatrix64F.wrap;

/**
 * @author Marc A. Suchard
 * @author Paul Bastide
 */

public class SafeMultivariatePositiveSemidefiniteActualizedWithDriftIntegrator extends SafeMultivariateDiagonalActualizedWithDriftIntegrator {

    private static boolean DEBUG = false;

    public SafeMultivariatePositiveSemidefiniteActualizedWithDriftIntegrator(PrecisionType precisionType, int numTraits, int dimTrait, int bufferCount,
                                                         int diffusionCount) {
        super(precisionType, numTraits, dimTrait, bufferCount, diffusionCount);

        System.err.println("Trying SafeMultivariatePositiveSemidefiniteActualizedWithDriftIntegrator");
    }

    @Override
    public void transformDiffusionPrecision(int precisionIndex, DenseMatrix64F V) {

        // Precision
        double[] matrix = new double[dimTrait * dimTrait];
        System.arraycopy(diffusions, dimTrait * dimTrait * precisionIndex, matrix, 0, dimTrait * dimTrait);

        DenseMatrix64F P = wrap(dimTrait, dimTrait, matrix);
        DenseMatrix64F tmp = new DenseMatrix64F(dimTrait, dimTrait);
        CommonOps.multTransA(V, P, tmp);
        CommonOps.mult(tmp, V, P);
        unwrap(P, diffusions, dimTrait * dimTrait * precisionIndex);

        // Variance
        DenseMatrix64F variance = new DenseMatrix64F(dimTrait, dimTrait);
        CommonOps.invert(P, variance);
        unwrap(variance, inverseDiffusions, dimTrait * dimTrait * precisionIndex);
    }

//    @Override
//    public void transformPrior(int rootBufferIndex, int priorBufferIndex, DenseMatrix64F V) {
//
//        int priorOffset = dimPartial * priorBufferIndex;
////        int rootOffset = dimPartial * rootBufferIndex;
//
//        final DenseMatrix64F muPrior = MissingOps.wrap(partials, priorOffset , dimTrait, 1);
//        DenseMatrix64F tmpVec = new DenseMatrix64F(dimTrait, 1);
//        CommonOps.multTransA(V, muPrior, tmpVec);
//        unwrap(tmpVec, partials, priorOffset);
//
////        final DenseMatrix64F PRoot = MissingOps.wrap(partials, rootOffset + dimTrait, dimTrait, dimTrait);
////        final boolean useVariance = anyDiagonalInfinities(PRoot);
////        DenseMatrix64F tmpMat = new DenseMatrix64F(dimTrait, dimTrait);
////        if (useVariance) {
////            final DenseMatrix64F VPrior = MissingOps.wrap(partials, priorOffset + dimTrait + dimTrait * dimTrait, dimTrait, dimTrait);
////            CommonOps.multTransA(V, VPrior, tmpMat);
////            CommonOps.mult(tmpMat, V, VPrior);
////            unwrap(VPrior, partials, priorOffset+ dimTrait + dimTrait * dimTrait);
////        } else {
////            final DenseMatrix64F PPrior = MissingOps.wrap(partials, priorOffset + dimTrait, dimTrait, dimTrait);
////            CommonOps.multTransA(V, PPrior, tmpMat);
////            CommonOps.mult(tmpMat, V, PPrior);
////            unwrap(PPrior, partials, priorOffset+ dimTrait);
////        }
//    }

//    @Override
//    public void setPostOrderPartial(int bufferIndex, final double[] partial) {
//
//        // Transform partials
//
//
//
//        // Allocate
//        super.setPostOrderPartial(bufferIndex, partial);
//    }
//
//    @Override
//    public void getPostOrderPartial(int bufferIndex, final double[] partial) {
//        // Transform partials back
//
//        // get
//        super.getPostOrderPartial(bufferIndex, partial);
//    }

}
