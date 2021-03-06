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

import net.dedekind.lapack.Lapack;
import net.frobenius.lapack.PlainLapack;

/**
 * QR decomposition of a double-precision complex m-by-n, with {@code m >= n},
 * matrix {@code A} into the product {@code A = Q * R} where {@code Q} is a
 * m-by-n orthogonal matrix and {@code R} is a n-by-n upper triangular matrix.
 */
public final class QrdComplexD {

    // m x n orthogonal matrix Q
    private final ComplexMatrixD Q;
    // n x n upper triangular matrix R
    private final ComplexMatrixD R;

    /**
     * The {@code m x n} orthogonal matrix Q factor in the product
     * {@code A = Q * R}.
     * 
     * @return the orthogonal factor {@code Q} of the QR decomposition
     */
    public ComplexMatrixD getQ() {
        return Q;
    }

    /**
     * The {@code n x n} upper triangular matrix R factor in the product
     * {@code A = Q * R}.
     * 
     * @return the upper triangular factor {@code R} of the QR decomposition
     */
    public ComplexMatrixD getR() {
        return R;
    }

    /* package */ QrdComplexD(ComplexMatrixD A) {
        if (A.numRows() < A.numColumns()) {
            throw new IllegalArgumentException("QR decomposition only works for m x n matrices where"
                    + " m >= n. But this is a " + A.numRows() + " x " + A.numColumns() + " matrix");
        }
        Q = A.copy();
        R = new SimpleComplexMatrixD(A.numColumns(), A.numColumns());
        computeQrdInplace(Q);
    }

    private void computeQrdInplace(ComplexMatrixD AA) {
        int m = AA.numRows();
        int n = AA.numColumns();
        int k = Math.min(m, n);
        int lda = Math.max(1, m);
        double[] tau = new double[2 * k];
        Lapack la = Lapack.getInstance();
        // compute the QR factorization
        PlainLapack.zgeqrf(la, m, n, AA.getArrayUnsafe(), lda, tau);
        R.setInplaceUpperTrapezoidal(AA);
        // compute the elements of Q explicitly
        PlainLapack.zungqr(la, m, n, k, AA.getArrayUnsafe(), lda, tau);
    }
}
