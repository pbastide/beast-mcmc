/*
 * HomogenousDiffusionModelDelegate.java
 *
 * Copyright (c) 2002-2016 Alexei Drummond, Andrew Rambaut and Marc Suchard
 *
 * This file is part of BEAST.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership and licensing.
 *
 * BEAST is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 *  BEAST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAST; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301  USA
 */

package dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evomodel.branchratemodel.BranchRateModel;
import dr.evomodel.continuous.MultivariateDiffusionModel;
import dr.evomodel.treedatalikelihood.continuous.cdi.ContinuousDiffusionIntegrator;
import dr.evomodel.treedatalikelihood.continuous.cdi.SafeMultivariatePositiveSemidefiniteActualizedWithDriftIntegrator;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.Model;
import dr.math.matrixAlgebra.missingData.MissingOps;
import org.ejml.data.Complex64F;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.interfaces.decomposition.EigenDecomposition;
import org.ejml.ops.CommonOps;
import org.ejml.ops.EigenOps;
import org.ejml.ops.MatrixFeatures;

import static dr.math.matrixAlgebra.missingData.MissingOps.*;
import static org.ejml.data.DenseMatrix64F.wrap;

import java.util.List;

/**
 * A simple OU diffusion model delegate with branch-specific drift and constant diffusion
 * @author Marc A. Suchard
 * @author Paul Bastide
 * @version $Id$
 */
public final class PositiveSemidefiniteOrnsteinUhlenbeckDiffusionModelDelegate extends AbstractDiffusionModelDelegate {

    // Here, branchRateModels represents optimal values

    private final int dim;
    private final List<BranchRateModel> branchRateModels;

    protected CompoundSymmetricMatrix strengthOfSelectionMatrixParameter;
//    private double[][] strengthOfSelectionMatrix;
    protected EigenDecomposition eigenDecompositionStrengthOfSelection;


    public PositiveSemidefiniteOrnsteinUhlenbeckDiffusionModelDelegate(Tree tree,
                                                                       MultivariateDiffusionModel diffusionModel,
                                                                       List<BranchRateModel> branchRateModels,
                                                                       CompoundSymmetricMatrix strengthOfSelectionMatrixParam) {
        this(tree, diffusionModel, branchRateModels, strengthOfSelectionMatrixParam, 0);
    }

    private PositiveSemidefiniteOrnsteinUhlenbeckDiffusionModelDelegate(Tree tree,
                                                                        MultivariateDiffusionModel diffusionModel,
                                                                        List<BranchRateModel> branchRateModels,
                                                                        CompoundSymmetricMatrix strengthOfSelectionMatrixParam,
                                                                        int partitionNumber) {
        super(tree, diffusionModel, partitionNumber);
        this.branchRateModels = branchRateModels;

        dim = diffusionModel.getPrecisionParameter().getColumnDimension();

        if (branchRateModels != null) {

            for (BranchRateModel rateModel : branchRateModels) {
                addModel(rateModel);
            }

            if (branchRateModels.size() != dim) {
                throw new IllegalArgumentException("Invalid dimensions");
            }
        }

        // Strength of selection matrix
        this.strengthOfSelectionMatrixParameter = strengthOfSelectionMatrixParam;
        addVariable(strengthOfSelectionMatrixParameter);
        // Eigen decomposition
        this.eigenDecompositionStrengthOfSelection = decomposeStrenghtOfSelection(strengthOfSelectionMatrixParam);

    }

    private EigenDecomposition decomposeStrenghtOfSelection(CompoundSymmetricMatrix Asym){
        DenseMatrix64F A = MissingOps.wrap(Asym);
        int n = A.numCols;
        // Checks
        if (n != A.numRows) throw new RuntimeException("Selection strength A matrix must be square.");
        if (!MatrixFeatures.isSymmetric(A)) throw new RuntimeException("Selection strength A matrix must be symmetric.");
        // Decomposition
        EigenDecomposition eigA = DecompositionFactory.eig(n, true, true);
        if( !eigA.decompose(A) ) throw new RuntimeException("Eigen decomposition failed.");
        return eigA;
    }


    public double[][] getStrengthOfSelection() {
        return strengthOfSelectionMatrixParameter.getParameterAsMatrix();
    }

    public double[] getEigenValuesStrengthOfSelection() {
        double[] eigA = new double[dim];
        for (int p = 0; p < dim; ++p) {
            Complex64F ev = eigenDecompositionStrengthOfSelection.getEigenvalue(p);
            if (!ev.isReal()) throw new RuntimeException("Selection strength A should only have real eigenvalues.");
            eigA[p] = ev.real;
        }
        return eigA;
    }

    public DenseMatrix64F getEigenVectorsStrengthOfSelection() {
        return EigenOps.createMatrixV(eigenDecompositionStrengthOfSelection);
    }

    @Override
    protected void handleModelChangedEvent(Model model, Object object, int index) {

        if (branchRateModels.contains(model)) {
            fireModelChanged(model);
        } else {
            super.handleModelChangedEvent(model, object, index);
        }
    }

    @Override
    public boolean hasDrift() { return true; }

    @Override
    public boolean hasActualization() { return true; }

    @Override
    protected double[] getDriftRates(int[] branchIndices, int updateCount) {

        final double[] drift = new double[updateCount * dim];  // TODO Reuse?

        if (branchRateModels != null) {

            int offset = 0;
            for (int i = 0; i < updateCount; ++i) {

                final NodeRef node = tree.getNode(branchIndices[i]); // TODO Check if correct node
                double[] driftNode = new double[dim];


                for (int model = 0; model < dim; ++model) {
                    driftNode[model] = branchRateModels.get(model).getBranchRate(tree, node);
                }
                transformVector(driftNode);
                for (int model = 0; model < dim; ++model) {
                    drift[offset] = driftNode[model];
                    ++offset;
                }
            }
        }

        return drift;
    }

    private void transformVector(double[] vector) {

        assert(vector.length == dim);

        DenseMatrix64F B = wrap(dim, 1, vector);
        DenseMatrix64F V = getEigenVectorsStrengthOfSelection();
        DenseMatrix64F tmp = new DenseMatrix64F(dim, 1);
        CommonOps.multTransA(V, B, tmp);

        unwrap(tmp, vector, 0);
    }

    private void transformVectorBack(double[] vector) {

        assert(vector.length == dim);

        DenseMatrix64F B = wrap(dim, 1, vector);
        DenseMatrix64F V = getEigenVectorsStrengthOfSelection();
        DenseMatrix64F tmp = new DenseMatrix64F(dim, 1);
        CommonOps.mult(V, B, tmp);

        unwrap(tmp, vector, 0);
    }

    private void transformVectorBack(DenseMatrix64F B) {

        assert(B.numRows == dim);
        assert(B.numCols == 1);

        DenseMatrix64F V = getEigenVectorsStrengthOfSelection();
        DenseMatrix64F tmp = new DenseMatrix64F(dim, 1);
        CommonOps.mult(V, B, tmp);
        B.set(tmp);
    }

//    protected void transformPrecision(double[][] matrix) {
//
//        assert(matrix.length == dim * dim);
//
//        DenseMatrix64F P = new DenseMatrix64F(dim, dim);
//        for (int i = 0; i < dim; i++) {
//            for (int j = 0; j < dim; j++) {
//                P.set(i,j, matrix[i][j]);
//            }
//        }
//        DenseMatrix64F V = getEigenVectorsStrengthOfSelection();
//
//        DenseMatrix64F tmp = new DenseMatrix64F(dim, dim);
//        CommonOps.mult(V, P, tmp);
//        CommonOps.multTransB(tmp, V, P);
//
//        for (int i = 0; i < dim; i++) {
//            for (int j = 0; j < dim; j++) {
//                matrix[i][j] = P.get(i,j);
//            }
//        }
//
//    }

//    protected void transformData(double[] data, int nTips) {
//
//        double[] vector = new double[dim];
//
//        for (int i = 0; i < nTips; i++) {
//            System.arraycopy(data, i*dim, vector, 0, dim);
//            naToZeros(vector);
//            transformVector(vector);
//            System.arraycopy(vector, 0, data, i*dim, dim);
//        }
//    }

//    private void naToZeros(double[] vector) {
//        for (int i = 0; i < vector.length; i++) {
//            if (Double.isNaN(vector[i])) {
//                vector[i] = 0.0;
//            }
//        }
//    }

    @Override
    public void transformTipPartial(double[] tipPartial){
        transformPartial(tipPartial, 0);
    }

    private void transformPartial(double[] tipPartial, int offset){
        double[] data = new double[dim];
        System.arraycopy(tipPartial, offset, data, 0, dim);
        transformVector(data);
        System.arraycopy(data, 0, tipPartial, offset, dim);
    }

//    @Override
//    public void transformPrior(int rootBufferIndex, int priorBufferIndex, ContinuousDiffusionIntegrator cdi) {
//        cdi.transformPrior(rootBufferIndex, priorBufferIndex, getEigenVectorsStrengthOfSelection());
//    }

    @Override
    public void setDiffusionModels(ContinuousDiffusionIntegrator cdi, boolean flip) {
        super.setDiffusionModels(cdi, flip);

        assert(cdi instanceof SafeMultivariatePositiveSemidefiniteActualizedWithDriftIntegrator);

        cdi.transformDiffusionPrecision(getEigenBufferOffsetIndex(0), getEigenVectorsStrengthOfSelection());

        cdi.setDiffusionStationaryVariance(getEigenBufferOffsetIndex(0),
                getEigenValuesStrengthOfSelection());
    }

    @Override
    public void updateDiffusionMatrices(ContinuousDiffusionIntegrator cdi, int[] branchIndices, double[] edgeLengths,
                                        int updateCount, boolean flip) {

        int[] probabilityIndices = new int[updateCount];

        for (int i = 0; i < updateCount; i++) {
            if (flip) {
                flipMatrixBufferOffset(branchIndices[i]);
            }
            probabilityIndices[i] = getMatrixBufferOffsetIndex(branchIndices[i]);
        }

        cdi.updateOrnsteinUhlenbeckDiffusionMatrices(
                getEigenBufferOffsetIndex(0),
                probabilityIndices,
                edgeLengths,
                getDriftRates(branchIndices, updateCount),
                getEigenValuesStrengthOfSelection(),
                updateCount);
    }

    @Override
    public void transformRootPartial(ContinuousDiffusionIntegrator cdi, int priorBufferIndex, int numTraits, int dimPrecision) {
        final int length = dim + dimPrecision;
        double[] partial = new double [length * numTraits];
        cdi.getPostOrderPartial(priorBufferIndex, partial);

        int offset = 0;
        for (int trait = 0; trait < numTraits; ++trait) {
            transformPartial(partial, offset);
            offset += length;
        }

        cdi.setPostOrderPartial(priorBufferIndex, partial);
    }

    double[] getAccumulativeDrift(final NodeRef node, double[] priorMean) {
        transformVector(priorMean);
        final DenseMatrix64F drift = new DenseMatrix64F(dim, 1, true, priorMean);
        recursivelyAccumulateDrift(node, drift);
        transformVectorBack(drift);
        transformVectorBack(priorMean);
        return drift.data;
    }

    private void recursivelyAccumulateDrift(final NodeRef node, final DenseMatrix64F drift) {
        if (!tree.isRoot(node)) {

            // Compute parent
            recursivelyAccumulateDrift(tree.getParent(node), drift);

            // Actualize
            int[] branchIndice = new int[1];
            branchIndice[0] = getMatrixBufferOffsetIndex(node.getNumber());

            final double length = tree.getBranchLength(node);

            double[] actualization = new double[dim];
            computeActualizationBranch(-length, actualization);

            for (int p = 0; p < dim; ++p) {
                drift.set(p, 0, actualization[p] * drift.get(p, 0));
            }

            // Add optimal value
            double[] optVal = getDriftRates(branchIndice, 1);

            for (int p = 0; p < dim; ++p) {
                drift.set(p, 0, drift.get(p, 0) + (1 - actualization[p]) *  optVal[p]);
            }

        }
    }

    private void computeActualizationBranch(double lambda, double[] C){
        double[] A = getEigenValuesStrengthOfSelection();
        for (int p = 0; p < dim; ++p) {
            C[p] = Math.exp(lambda * A[p]);
        }
    }
}