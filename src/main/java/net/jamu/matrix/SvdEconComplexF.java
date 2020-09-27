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
 * Economy singular value decomposition of a single-precision complex m-by-n
 * matrix {@code A}.
 * <p>
 * The economy-size decomposition removes extra rows or columns of zeros from
 * the diagonal matrix of singular values, {@code S}, along with the columns in
 * either {@code U} or {@code V} that multiply those zeros in the expression
 * {@code A = U * S * congugate-transpose(V)}. Removing these zeros and columns
 * can improve execution time and reduce storage requirements without
 * compromising the accuracy of the decomposition.
 */
public final class SvdEconComplexF extends SvdComplexF {

    /**
     * The left singular vectors (column-wise).
     * 
     * @return reduced m-by-r semi-unitary matrix {@code U} where {@code r} is
     *         the rank of {@code A}
     */
    public ComplexMatrixF getU() {
        return U;
    }

    /**
     * The right singular vectors (row-wise).
     * <p>
     * Note that the algorithm returns <code>V<sup>*</sup></code>, (i.e., the
     * conjugate transpose), not {@code V}.
     * 
     * @return reduced n-by-r semi-unitary matrix <code>V<sup>*</sup></code>
     *         where {@code r} is the rank of {@code A}
     */
    public ComplexMatrixF getVh() {
        return Vh;
    }

    /**
     * The non-zero singular values in descending order.
     * 
     * @return array of size {@code r <= min(m, n)} containing the non-zero
     *         singular values in descending order
     */
    public float[] getS() {
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

    /* package */ SvdEconComplexF(ComplexMatrixF A) {
        // jobType, U, Vh, S
        super(TSvdJob.PART, new SimpleComplexMatrixF(Math.max(1, A.numRows()), Math.min(A.numRows(), A.numColumns())),
                new SimpleComplexMatrixF(Math.max(1, Math.min(A.numRows(), A.numColumns())), A.numColumns()),
                new float[Math.min(A.numRows(), A.numColumns())]);
        computeSvdInplace(A);
    }

    private void computeSvdInplace(ComplexMatrixF A) {
        // The only case where A must be copied before calling the complex
        // '?gesdd' is when jobz == 'O' (TSvdJob.OVERWRITE) which we never use
        // here
        int m = A.numRows();
        int n = A.numColumns();
        PlainLapack.cgesdd(Lapack.getInstance(), jobType, m, n, A.getArrayUnsafe(), Math.max(1, m), S,
                U.getArrayUnsafe(), Math.max(1, U.numRows()), Vh.getArrayUnsafe(), Math.max(1, Vh.numRows()));
    }
}
