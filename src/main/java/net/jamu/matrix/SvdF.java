/*
 * Copyright 2019, 2020 Stefan Zobel
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

import net.frobenius.TSvdJob;
import net.frobenius.lapack.PlainLapack;

/**
 * Singular value decomposition of a float m-by-n matrix {@code A}.
 * <p>
 * The SVD is written
 * 
 * <pre>
 * {@code
 * A = U * S * transpose(V)
 * }
 * </pre>
 *
 * where {@code S} is an m-by-n matrix which is zero except for its min(m, n)
 * diagonal elements, {@code U} is an m-by-m orthogonal matrix, and {@code V} is
 * a n-by-n orthogonal matrix. The diagonal elements of {@code S} are the
 * singular values of {@code A}; they are real and non-negative, and are
 * returned in descending order. The first min(m, n) columns of {@code U} and
 * {@code V} are the left and right singular vectors of {@code A}.
 */
public class SvdF {

    // either "A", "N" or "S"
    protected final TSvdJob jobType;

    protected final SimpleMatrixF U;
    protected final SimpleMatrixF Vt;
    protected final float[] S;

    /**
     * The left singular vectors (column-wise) or {@code null} if the singular
     * vectors haven't been computed.
     * 
     * @return m-by-m orthogonal matrix
     */
    public MatrixF getU() {
        return U;
    }

    /**
     * The right singular vectors (row-wise) or {@code null} if the singular
     * vectors haven't been computed.
     * <p>
     * Note that the algorithm returns <code>V<sup>T</sup></code>, not
     * {@code V}.
     * 
     * @return n-by-n orthogonal matrix
     */
    public MatrixF getVt() {
        return Vt;
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

    /* package */ SvdF(MatrixF A, boolean full) {
        S = new float[Math.min(A.numRows(), A.numColumns())];
        jobType = full ? TSvdJob.ALL : TSvdJob.NONE;
        if (jobType == TSvdJob.ALL) {
            U = new SimpleMatrixF(A.numRows(), A.numRows());
            Vt = new SimpleMatrixF(A.numColumns(), A.numColumns());
        } else {
            U = null;
            Vt = null;
        }
        computeSvdInplace(A);
    }

    /* package */ SvdF(TSvdJob jobType, SimpleMatrixF U, SimpleMatrixF Vt, float[] S) {
        this.jobType = jobType;
        this.U = U;
        this.Vt = Vt;
        this.S = S;
    }

    private void computeSvdInplace(MatrixF A) {
        // Note: this wouldn't work for TSvdJob.OVERWRITE as A gets overwritten
        // in that case
        MatrixF AA = A.copy();
        int m = AA.numRows();
        int n = AA.numColumns();
        PlainLapack.sgesdd(Matrices.getLapack(), jobType, m, n, AA.getArrayUnsafe(), Math.max(1, m), S,
                hasSingularVectors() ? U.getArrayUnsafe() : new float[0], Math.max(1, m),
                hasSingularVectors() ? Vt.getArrayUnsafe() : new float[0], Math.max(1, n));
    }
}
