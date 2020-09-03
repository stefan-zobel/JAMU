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
 * Economy singular value decomposition of a double m-by-n matrix {@code A}.
 * <p>
 * The economy-size decomposition removes extra rows or columns of zeros from
 * the diagonal matrix of singular values, {@code S}, along with the columns in
 * either {@code U} or {@code V} that multiply those zeros in the expression
 * {@code A = U * S * transpose(V)}. Removing these zeros and columns can
 * improve execution time and reduce storage requirements without compromising
 * the accuracy of the decomposition.
 */
public class SvdEconD extends SvdD {

    /**
     * The left singular vectors (column-wise).
     * 
     * @return reduced m-by-m semi-orthogonal matrix
     */
    public MatrixD getU() {
        return U;
    }

    /**
     * The right singular vectors (row-wise).
     * <p>
     * Note that the algorithm returns <code>V<sup>T</sup></code>, not
     * {@code V}.
     * 
     * @return reduced n-by-n semi-orthogonal matrix
     */
    public MatrixD getVt() {
        return Vt;
    }

    /**
     * The non-zero singular values in descending order.
     * 
     * @return array of size {@code min(m, n)} containing the non-zero singular
     *         values in descending order
     */
    public double[] getS() {
        return S;
    }

    /**
     * Always {@code true} as the singular vectors will always be computed.
     * 
     * @return whether singular vectors have been computed or not
     */
    public boolean hasSingularVectors() {
        return true;
    }

    /* package */ SvdEconD(MatrixD A) {
        // jobType, U, Vt, S
        super(TSvdJob.PART, new SimpleMatrixD(Math.max(1, A.numRows()), Math.min(A.numRows(), A.numColumns())),
                new SimpleMatrixD(Math.max(1, Math.min(A.numRows(), A.numColumns())), A.numColumns()),
                new double[Math.min(A.numRows(), A.numColumns())]);
        computeSvdInplace(A);
    }

    private void computeSvdInplace(MatrixD A) {
        // Note: this wouldn't work for TSvdJob.OVERWRITE as A gets overwritten
        // in that case
        MatrixD AA = A.copy();
        int m = AA.numRows();
        int n = AA.numColumns();
        PlainLapack.dgesdd(Lapack.getInstance(), jobType, m, n, AA.getArrayUnsafe(), Math.max(1, m), S,
                U.getArrayUnsafe(), Math.max(1, U.numRows()), Vt.getArrayUnsafe(), Math.max(1, Vt.numRows()));
    }
}
