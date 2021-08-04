/*
 * Copyright 2020, 2021 Stefan Zobel
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

import net.dedekind.lapack.Lapack;
import net.frobenius.ComputationTruncatedException;
import net.frobenius.lapack.PlainLapack;
import net.jamu.complex.ZfImpl;

/**
 * LU decomposition of a general m-by-n single precision complex matrix
 * {@code A = P * L * U} using partial pivoting with row interchanges where
 * {@code P} is a permutation matrix, {@code L} is a lower triangular (lower
 * trapezoidal if m > n) matrix with unit diagonal elements and {@code U} is an
 * upper triangular (upper trapezoidal if m < n) matrix.
 * <p>
 * Note that this implementation returns {@code null} for {@code P} when no row
 * interchanges are necessary (in that case, if the caller wants to consistently
 * use the {@code A = P * L * U} formulation (instead of the simpler
 * {@code A = L * U}), simply use the identity matrix for {@code P}).
 */
public final class LudComplexF {

    // true if matrix A was singular
    private boolean isSingular;

    // the pivot vector as returned by LAPACK
    // (keep that for the case that we need to call other LAPACK routines later)
    private final int[] pivot;

    // permutation matrix or null if no permutations are necessary
    private final ComplexMatrixF P;

    // lower triangular (lower trapezoidal if m > n) with unit diagonal elements
    private final ComplexMatrixF L;

    // upper triangular (upper trapezoidal if m < n)
    private final ComplexMatrixF U;

    /**
     * If row interchanges are needed returns the (quadratic) permutation matrix
     * {@code P}, otherwise {@code null} is returned. In the {@code null} case,
     * if the caller wants to consistently use the {@code A = P * L * U}
     * formulation (instead of the simpler {@code A = L * U}), simply use the
     * identity matrix for {@code P}.
     * 
     * @return {@code null} when no row interchanges are necessary, otherwise
     *         the permutation matrix that describes the row interchanges
     */
    public ComplexMatrixF getP() {
        return P;
    }

    /**
     * The lower triangular (lower trapezoidal if m > n) factor {@code L} in the
     * product {@code A = P * L * U}.
     * 
     * @return the lower triangular (lower trapezoidal if m > n) factor
     *         {@code L} of the {@code LU} decomposition
     */
    public ComplexMatrixF getL() {
        return L;
    }

    /**
     * The factor {@code P * L} in the product {@code A = P * L * U}.
     * 
     * @return the factor {@code P * L} of the {@code PLU} decomposition
     * @since 1.2
     */
    public ComplexMatrixF getPL() {
        return (P == null) ? L : P.times(L);
    }

    /**
     * The upper triangular (upper trapezoidal if m < n) factor {@code U} in the
     * product {@code A = P * L * U}.
     * 
     * @return the upper triangular (upper trapezoidal if m < n) factor
     *         {@code U} of the {@code LU} decomposition
     */
    public ComplexMatrixF getU() {
        return U;
    }

    /**
     * Returns {@code true} when the matrix {@code A} was singular, otherwise
     * {@code false}.
     * 
     * @return {@code true} if {@code A} is singular, otherwise {@code false}
     */
    public boolean isSingular() {
        return isSingular;
    }

    /* package */ LudComplexF(ComplexMatrixF A) {
        int m = A.numRows();
        int n = A.numColumns();
        pivot = new int[Math.min(m, n)];
        if (m >= n) {
            L = Matrices.createComplexF(m, n);
            U = Matrices.createComplexF(n, n);
        } else {
            L = Matrices.createComplexF(m, m);
            U = Matrices.createComplexF(m, n);
        }
        P = computeLudInplace(A, L.numRows());
    }

    private ComplexMatrixF computeLudInplace(ComplexMatrixF A, int dimP) {
        ComplexMatrixF AA = A.copy();
        try {
            int lda = Math.max(1, AA.numRows());
            PlainLapack.cgetrf(Lapack.getInstance(), AA.numRows(), AA.numColumns(), AA.getArrayUnsafe(), lda, pivot);
        } catch (ComputationTruncatedException e) {
            isSingular = true;
        }
        copyIntoL(AA);
        copyIntoU(AA);
        // return P
        return Permutation.genPermutationComplexMatrixF(pivot, dimP);
    }

    private void copyIntoL(ComplexMatrixF AA) {
        ComplexMatrixF l_ = L;
        int cols = l_.numColumns();
        int rows = l_.numRows();
        ZfImpl z = new ZfImpl(0.0f);
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                if (row == col) {
                    l_.set(row, col, 1.0f, 1.0f);
                } else if (row > col) {
                    AA.getUnsafe(row, col, z);
                    l_.set(row, col, z.re(), z.im());
                }
            }
        }
    }

    private void copyIntoU(ComplexMatrixF AA) {
        ComplexMatrixF u_ = U;
        int cols = u_.numColumns();
        int rows = u_.numRows();
        ZfImpl z = new ZfImpl(0.0f);
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                if (row <= col) {
                    AA.getUnsafe(row, col, z);
                    u_.set(row, col, z.re(), z.im());
                }
            }
        }
    }
}
