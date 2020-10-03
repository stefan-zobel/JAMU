/*
 * Copyright 2020 Stefan Zobel
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.jamu.matrix;

/**
 * Build a <a href=https://en.wikipedia.org/wiki/Permutation_matrix>permutation
 * matrix</a> from an array of pivot indexes (describing row interchanges) which
 * gets returned by LAPACK methods like {@code ?getrf}.
 * <p>
 * For example, an {@code 'ipiv'} array {@code [2,2,2]} in the {@code 3 x 3}
 * case means that row 1 was interchanged with row 2, then row 2 (which is the
 * original row 1) stayed the same, then row 3 was interchanged with row 2
 * (which was the original row 1).
 */
final class Permutation {

    /**
     * Returns a {@code (dim x dim)} quadratic permutation matrix when
     * {@code partialPivot} specifies at least one actual interchange of rows,
     * otherwise returns {@code null}.
     * 
     * @param partialPivot
     *            the {@code 'ipiv'} partial pivoting indexes returned by
     *            {@code ?getrf}
     * @param dim
     *            the dimension of the quadratic {@code (dim x dim)} permutation
     *            matrix
     * @return {@code null} when {@code partialPivot} doesn't describe any
     *         actual row interchanges, otherwise a full-fledged permutation
     *         matrix gets returned
     */
    static MatrixD genPermutationMatrixD(int[] partialPivot, int dim) {
        int[] permutation = checkPivotVector(partialPivot);
        if (permutation != null) {
            return buildMatrixD(permutation, dim);
        }
        return null;
    }

    /**
     * Returns a {@code (dim x dim)} quadratic permutation matrix when
     * {@code partialPivot} specifies at least one actual interchange of rows,
     * otherwise returns {@code null}.
     * 
     * @param partialPivot
     *            the {@code 'ipiv'} partial pivoting indexes returned by
     *            {@code ?getrf}
     * @param dim
     *            the dimension of the quadratic {@code (dim x dim)} permutation
     *            matrix
     * @return {@code null} when {@code partialPivot} doesn't describe any
     *         actual row interchanges, otherwise a full-fledged permutation
     *         matrix gets returned
     */
    static MatrixF genPermutationMatrixF(int[] partialPivot, int dim) {
        int[] permutation = checkPivotVector(partialPivot);
        if (permutation != null) {
            return buildMatrixF(permutation, dim);
        }
        return null;
    }

    /**
     * Returns a {@code (dim x dim)} quadratic permutation matrix when
     * {@code partialPivot} specifies at least one actual interchange of rows,
     * otherwise returns {@code null}.
     * 
     * @param partialPivot
     *            the {@code 'ipiv'} partial pivoting indexes returned by
     *            {@code ?getrf}
     * @param dim
     *            the dimension of the quadratic {@code (dim x dim)} permutation
     *            matrix
     * @return {@code null} when {@code partialPivot} doesn't describe any
     *         actual row interchanges, otherwise a full-fledged permutation
     *         matrix gets returned
     */
    static ComplexMatrixD genPermutationComplexMatrixD(int[] partialPivot, int dim) {
        int[] permutation = checkPivotVector(partialPivot);
        if (permutation != null) {
            return buildComplexMatrixD(permutation, dim);
        }
        return null;
    }

    /**
     * Returns a {@code (dim x dim)} quadratic permutation matrix when
     * {@code partialPivot} specifies at least one actual interchange of rows,
     * otherwise returns {@code null}.
     * 
     * @param partialPivot
     *            the {@code 'ipiv'} partial pivoting indexes returned by
     *            {@code ?getrf}
     * @param dim
     *            the dimension of the quadratic {@code (dim x dim)} permutation
     *            matrix
     * @return {@code null} when {@code partialPivot} doesn't describe any
     *         actual row interchanges, otherwise a full-fledged permutation
     *         matrix gets returned
     */
    static ComplexMatrixF genPermutationComplexMatrixF(int[] partialPivot, int dim) {
        int[] permutation = checkPivotVector(partialPivot);
        if (permutation != null) {
            return buildComplexMatrixF(permutation, dim);
        }
        return null;
    }

    private static MatrixD buildMatrixD(int[] perm, int dim) {
        int n = perm.length;
        MatrixD permMatrix = Matrices.createD(dim, dim);
        for (int i = 0; i < dim; ++i) {
            if (i < n) {
                permMatrix.set(perm[i] - 1, i, 1.0);
            } else {
                permMatrix.set(i, i, 1.0);
            }
        }
        return permMatrix;
    }

    private static MatrixF buildMatrixF(int[] perm, int dim) {
        int n = perm.length;
        MatrixF permMatrix = Matrices.createF(dim, dim);
        for (int i = 0; i < dim; ++i) {
            if (i < n) {
                permMatrix.set(perm[i] - 1, i, 1.0f);
            } else {
                permMatrix.set(i, i, 1.0f);
            }
        }
        return permMatrix;
    }

    private static ComplexMatrixD buildComplexMatrixD(int[] perm, int dim) {
        int n = perm.length;
        ComplexMatrixD permMatrix = Matrices.createComplexD(dim, dim);
        for (int i = 0; i < dim; ++i) {
            if (i < n) {
                permMatrix.set(perm[i] - 1, i, 1.0, 1.0);
            } else {
                permMatrix.set(i, i, 1.0, 1.0);
            }
        }
        return permMatrix;
    }

    private static ComplexMatrixF buildComplexMatrixF(int[] perm, int dim) {
        int n = perm.length;
        ComplexMatrixF permMatrix = Matrices.createComplexF(dim, dim);
        for (int i = 0; i < dim; ++i) {
            if (i < n) {
                permMatrix.set(perm[i] - 1, i, 1.0f, 1.0f);
            } else {
                permMatrix.set(i, i, 1.0f, 1.0f);
            }
        }
        return permMatrix;
    }

    private static int[] checkPivotVector(int[] pivot) {
        if (pivot != null && pivot.length > 0) {
            boolean rowSwapDetected = false;
            for (int i = 0; i < pivot.length; ++i) {
                if (pivot[i] != i + 1) {
                    rowSwapDetected = true;
                    break;
                }
            }
            if (rowSwapDetected) {
                int[] perm = initPermVector(pivot);
                boolean positionsChanged = swapRows(pivot, perm);
                if (positionsChanged) {
                    return perm;
                }
            }
        }
        return null;
    }

    private static boolean swapRows(int[] pivot, int[] perm) {
        boolean swapped = false;
        for (int j = 0; j < pivot.length; ++j) {
            int rowIndex = j + 1;
            int newRow = pivot[j];
            if (newRow != rowIndex) {
                int oldRow = perm[rowIndex - 1];
                perm[rowIndex - 1] = perm[newRow - 1];
                perm[newRow - 1] = oldRow;
                swapped = true;
            }
        }
        if (swapped) {
            for (int i = 0; i < perm.length; ++i) {
                if (perm[i] != i + 1) {
                    return true;
                }
            }
        }
        return false;
    }

    private static int[] initPermVector(int[] pivot) {
        int[] perm = new int[Math.max(pivot.length, getMaxRowIndex(pivot))];
        for (int i = 0; i < perm.length; ++i) {
            perm[i] = i + 1;
        }
        return perm;
    }

    private static int getMaxRowIndex(int[] pivot) {
        int maxRowIdx = -1;
        for (int i = 0; i < pivot.length; ++i) {
            int row = pivot[i];
            if (row > maxRowIdx) {
                maxRowIdx = row;
            }
        }
        return maxRowIdx;
    }

    private Permutation() {
        throw new AssertionError();
    }
}
