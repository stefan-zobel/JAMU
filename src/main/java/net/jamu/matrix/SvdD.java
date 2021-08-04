/*
 * Copyright 2019, 2021 Stefan Zobel
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
 * Singular value decomposition of a double m-by-n matrix {@code A}.
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
public class SvdD {

    // either "A", "N" or "S"
    protected final TSvdJob jobType;

    protected final SimpleMatrixD U;
    protected final SimpleMatrixD Vt;
    protected final double[] S;

    /**
     * Computes the approximately optimal Singular Value truncation ("Singular
     * Values Hard Threshold (SVHT)") after <a
     * href=https://arxiv.org/pdf/1305.5870.pdf>Gavish and Donoho (2014)</a> for
     * a real matrix {@code A} of dimension {@code rows x cols} and its
     * corresponding (economy) SVD decomposition {@code svd}.
     * 
     * @param rows
     *            number of rows of matrix {@code A}
     * @param cols
     *            number of columns of matrix {@code A}
     * @param svd
     *            the SVD (or economy SVD) of matrix {@code A}
     * @return the approximately optimal Singular Value truncation value as per
     *         Gavish and Donoho
     * @see "Gavish, M., & Donoho, D.L. (2014). The optimal hard threshold for singular values is 4/sqrt(3). IEEE Transactions on Information Theory, 60(8), 5040-5053."
     * @since 1.1
     */
    public static int optimalHardThreshold(int rows, int cols, SvdD svd) {
        return SVHT.threshold(rows, cols, svd.getS());
    }

    /**
     * The left singular vectors (column-wise) or {@code null} if the singular
     * vectors haven't been computed.
     * 
     * @return m-by-m orthogonal matrix
     */
    public MatrixD getU() {
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
    public MatrixD getVt() {
        return Vt;
    }

    /**
     * The singular values in descending order.
     * 
     * @return array containing the singular values in descending order
     */
    public double[] getS() {
        return S;
    }

    /* package */ double norm2() {
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

    /* package */ SvdD(MatrixD A, boolean full) {
        S = new double[Math.min(A.numRows(), A.numColumns())];
        jobType = full ? TSvdJob.ALL : TSvdJob.NONE;
        if (jobType == TSvdJob.ALL) {
            U = new SimpleMatrixD(A.numRows(), A.numRows());
            Vt = new SimpleMatrixD(A.numColumns(), A.numColumns());
        } else {
            U = null;
            Vt = null;
        }
        computeSvdInplace(A);
    }

    /* package */ SvdD(TSvdJob jobType, SimpleMatrixD U, SimpleMatrixD Vt, double[] S) {
        this.jobType = jobType;
        this.U = U;
        this.Vt = Vt;
        this.S = S;
    }

    private void computeSvdInplace(MatrixD A) {
        // Note: this wouldn't work for TSvdJob.OVERWRITE as A gets overwritten
        // in that case
        MatrixD AA = A.copy();
        int m = AA.numRows();
        int n = AA.numColumns();
        PlainLapack.dgesdd(Matrices.getLapack(), jobType, m, n, AA.getArrayUnsafe(), Math.max(1, m), S,
                hasSingularVectors() ? U.getArrayUnsafe() : new double[0], Math.max(1, m),
                hasSingularVectors() ? Vt.getArrayUnsafe() : new double[0], Math.max(1, n));
    }
}
