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
import net.frobenius.TSvdJob;
import net.frobenius.lapack.PlainLapack;

/**
 * Singular value decomposition of a single-precision complex m-by-n matrix
 * {@code A}.
 * <p>
 * The SVD is written
 * 
 * <pre>
 * {@code
 * A = U * S * conjugate-transpose(V)
 * }
 * </pre>
 *
 * where {@code S} is an m-by-n matrix which is zero except for its min(m, n)
 * diagonal elements, {@code U} is an m-by-m unitary matrix, and {@code V} is a
 * n-by-n unitary matrix. The diagonal elements of {@code S} are the singular
 * values of {@code A}; they are real and non-negative, and are returned in
 * descending order. The first min(m, n) columns of {@code U} and {@code V} are
 * the left and right singular vectors of {@code A}.
 */
public class SvdComplexF {

    // either "A", "N" or "S"
    protected final TSvdJob jobType;

    protected final SimpleComplexMatrixF U;
    protected final SimpleComplexMatrixF Vh;
    protected final float[] S;

    /**
     * The left singular vectors (column-wise) or {@code null} if the singular
     * vectors haven't been computed.
     * 
     * @return m-by-m unitary matrix {@code U}
     */
    public ComplexMatrixF getU() {
        return U;
    }

    /**
     * The right singular vectors (row-wise) or {@code null} if the singular
     * vectors haven't been computed.
     * <p>
     * Note that the algorithm returns <code>V<sup>*</sup></code> (i.e., the
     * conjugate transpose), not {@code V}.
     * 
     * @return n-by-n unitary matrix <code>V<sup>*</sup></code>
     */
    public ComplexMatrixF getVh() {
        return Vh;
    }

    /**
     * The singular values in descending order.
     * 
     * @return array containing the singular values in descending order
     */
    public float[] getS() {
        return S;
    }

    /* package */ float norm2() {
        return S[0];
    }

    /**
     * {@code true} if singular vectors have been computed, {@code false}
     * otherwise.
     * 
     * @return whether singular vectors have been computed or not
     */
    public boolean hasSingularVectors() {
        return jobType == TSvdJob.ALL;
    }

    /* package */ SvdComplexF(ComplexMatrixF A, boolean full) {
        S = new float[Math.min(A.numRows(), A.numColumns())];
        jobType = full ? TSvdJob.ALL : TSvdJob.NONE;
        if (jobType == TSvdJob.ALL) {
            U = new SimpleComplexMatrixF(A.numRows(), A.numRows());
            Vh = new SimpleComplexMatrixF(A.numColumns(), A.numColumns());
        } else {
            U = null;
            Vh = null;
        }
        computeSvdInplace(A);
    }

    /* package */ SvdComplexF(TSvdJob jobType, SimpleComplexMatrixF U, SimpleComplexMatrixF Vh, float[] S) {
        this.jobType = jobType;
        this.U = U;
        this.Vh = Vh;
        this.S = S;
    }

    private void computeSvdInplace(ComplexMatrixF A) {
        // The only case where A must be copied before calling the complex
        // '?gesdd' is when jobz == 'O' (TSvdJob.OVERWRITE) which we never use
        // here
        int m = A.numRows();
        int n = A.numColumns();
        PlainLapack.cgesdd(Lapack.getInstance(), jobType, m, n, A.getArrayUnsafe(), Math.max(1, m), S,
                hasSingularVectors() ? U.getArrayUnsafe() : new float[0], Math.max(1, m),
                hasSingularVectors() ? Vh.getArrayUnsafe() : new float[0], Math.max(1, n));
    }
}
