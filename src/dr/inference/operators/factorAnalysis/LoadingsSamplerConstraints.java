/*
 * LoadingsSamplerConstraints.java
 *
 * Copyright © 2002-2024 the BEAST Development Team
 * http://beast.community/about
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
 *
 */

package dr.inference.operators.factorAnalysis;


import java.util.ArrayList;

/**
 * @author Gabriel Hassler
 * @author Marc A. Suchard
 */

public interface LoadingsSamplerConstraints {

    int getColumnDim(int colIndex, int nRows);

    int getArrayIndex(int colIndex, int nRows);

    void allocateStorage(ArrayList<double[][]> precisionArray, ArrayList<double[]> midMeanArray,
                         ArrayList<double[]> meanArray, int nRows);


    enum ColumnDimProvider implements LoadingsSamplerConstraints {

        NONE("none") {
            @Override
            public int getColumnDim(int colIndex, int nRows) {
                return nRows;
            }

            @Override
            public int getArrayIndex(int colIndex, int nRows) {
                return 0;
            }

            @Override
            public void allocateStorage(ArrayList<double[][]> precisionArray, ArrayList<double[]> midMeanArray,
                                        ArrayList<double[]> meanArray, int nRows) {

                precisionArray.add(new double[nRows][nRows]);
                midMeanArray.add(new double[nRows]);
                meanArray.add(new double[nRows]);

            }
        },

        UPPER_TRIANGULAR("upperTriangular") {
            @Override
            public int getColumnDim(int colIndex, int nRows) {
                return Math.min(colIndex + 1, nRows);
            }

            @Override
            public int getArrayIndex(int colIndex, int nRows) {
                return Math.min(colIndex, nRows - 1);
            }

            @Override
            public void allocateStorage(ArrayList<double[][]> precisionArray, ArrayList<double[]> midMeanArray,
                                        ArrayList<double[]> meanArray, int nRows) {

                for (int i = 1; i <= nRows; i++) {
                    precisionArray.add(new double[i][i]);
                    midMeanArray.add(new double[i]);
                    meanArray.add(new double[i]);
                }

            }
        },

        FIRST_ROW("firstRow") {
            @Override
            public int getColumnDim(int colIndex, int nRows) {
                return 1;
            }

            @Override
            public int getArrayIndex(int colIndex, int nRows) {
                return 0;
            }

            @Override
            public void allocateStorage(ArrayList<double[][]> precisionArray, ArrayList<double[]> midMeanArray,
                                        ArrayList<double[]> meanArray, int nRows) {

                precisionArray.add(new double[1][1]);
                midMeanArray.add(new double[1]);
                meanArray.add(new double[1]);

            }
        },

        HYBRID("hybrid") {
            @Override
            public int getColumnDim(int colIndex, int nRows) {

                if (colIndex == 0) {
                    return 1;
                }
                return nRows;
            }

            @Override
            public int getArrayIndex(int colIndex, int nRows) {
                if (colIndex == 0) {
                    return 0;
                }
                return 1;
            }

            @Override
            public void allocateStorage(ArrayList<double[][]> precisionArray, ArrayList<double[]> midMeanArray, ArrayList<double[]> meanArray, int nRows) {

                // first column
                precisionArray.add(new double[1][1]);
                midMeanArray.add(new double[1]);
                meanArray.add(new double[1]);


                // remaining columns
                precisionArray.add(new double[nRows][nRows]);
                midMeanArray.add(new double[nRows]);
                meanArray.add(new double[nRows]);

            }
        };


        private static int[] convertToIndices(int i) {
            int[] indices = new int[i];
            for (int j = 0; j < i; j++) {
                indices[j] = j;
            }
            return indices;
        }

        public int[] getColumnIndices(int colIndex, int nRows) {
            return convertToIndices(getColumnDim(colIndex, nRows));
        }


        private String name;

        ColumnDimProvider(String name) {
            this.name = name;
        }

        public String getName() {
            return name;
        }

        public static ColumnDimProvider parse(String name) {
            name = name.toLowerCase();
            for (ColumnDimProvider dimProvider : ColumnDimProvider.values()) {
                if (name.compareTo(dimProvider.getName().toLowerCase()) == 0) {
                    return dimProvider;
                }
            }
            throw new IllegalArgumentException("Unknown dimension provider type");
        }

    }
}
