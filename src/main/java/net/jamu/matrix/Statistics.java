/*
 * Copyright 2021 Stefan Zobel
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
 * Some static utility methods for matrices that are useful in statistics.
 * 
 * @since 1.3
 */
public final class Statistics {

    /**
     * Creates a mean-centered copy of matrix {@code A}, i.e., the values in
     * each column of the copy have the mean of the corresponding column in
     * {@code A} subtracted creating a new matrix that is a de-meaned version of
     * {@code A}. Matrix {@code A} doesn't get mutated.
     * 
     * @param A
     *            the matrix that contains the non-centered data
     * @return a de-meaned copy of A
     */
    public static MatrixD center(MatrixD A) {
        return centerInplace(A.copy());
    }

    /**
     * Creates a mean-centered copy of matrix {@code A}, i.e., the values in
     * each column of the copy have the mean of the corresponding column in
     * {@code A} subtracted creating a new matrix that is a de-meaned version of
     * {@code A}. Matrix {@code A} doesn't get mutated.
     * 
     * @param A
     *            the matrix that contains the non-centered data
     * @return a de-meaned copy of A
     */
    public static MatrixF center(MatrixF A) {
        return centerInplace(A.copy());
    }

    /**
     * Creates a mean-centered copy of matrix {@code A}, i.e., the values in
     * each column of the copy have the mean of the corresponding column in
     * {@code A} subtracted creating a new matrix that is a de-meaned version of
     * {@code A}. Matrix {@code A} doesn't get mutated.
     * 
     * @param A
     *            the matrix that contains the non-centered data
     * @return a de-meaned copy of A
     */
    public static ComplexMatrixD center(ComplexMatrixD A) {
        return centerInplace(A.copy());
    }

    /**
     * Creates a mean-centered copy of matrix {@code A}, i.e., the values in
     * each column of the copy have the mean of the corresponding column in
     * {@code A} subtracted creating a new matrix that is a de-meaned version of
     * {@code A}. Matrix {@code A} doesn't get mutated.
     * 
     * @param A
     *            the matrix that contains the non-centered data
     * @return a de-meaned copy of A
     */
    public static ComplexMatrixF center(ComplexMatrixF A) {
        return centerInplace(A.copy());
    }

    /**
     * Subtracts the mean of each column {@code j} from each value in that
     * column {@code j}, effectively centering each column around its mean
     * (de-meaning that column so that its mean is zero). This is a destructive
     * operation that changes matrix {@code A} inplace.
     * 
     * @param A
     *            the matrix that contains the non-centered data
     * @return the matrix {@code A} de-meaned inplace
     */
    public static MatrixD centerInplace(MatrixD A) {
        double[] _a = A.getArrayUnsafe();
        int rows_ = A.numRows();
        int cols_ = A.numColumns();
        for (int col = 0; col < cols_; ++col) {
            double colSum = 0.0;
            double colMean = 0.0;
            for (int idx = col * rows_; idx < (col + 1) * rows_; ++idx) {
                colSum += _a[idx];
            }
            colMean = colSum / rows_;
            for (int idx = col * rows_; idx < (col + 1) * rows_; ++idx) {
                _a[idx] -= colMean;
            }
        }
        return A;
    }

    /**
     * Subtracts the mean of each column {@code j} from each value in that
     * column {@code j}, effectively centering each column around its mean
     * (de-meaning that column so that its mean is zero). This is a destructive
     * operation that changes matrix {@code A} inplace.
     * 
     * @param A
     *            the matrix that contains the non-centered data
     * @return the matrix {@code A} de-meaned inplace
     */
    public static MatrixF centerInplace(MatrixF A) {
        float[] _a = A.getArrayUnsafe();
        int rows_ = A.numRows();
        int cols_ = A.numColumns();
        for (int col = 0; col < cols_; ++col) {
            float colSum = 0.0f;
            float colMean = 0.0f;
            for (int idx = col * rows_; idx < (col + 1) * rows_; ++idx) {
                colSum += _a[idx];
            }
            colMean = colSum / rows_;
            for (int idx = col * rows_; idx < (col + 1) * rows_; ++idx) {
                _a[idx] -= colMean;
            }
        }
        return A;
    }

    /**
     * Subtracts the mean of each column {@code j} from each value in that
     * column {@code j}, effectively centering each column around its mean
     * (de-meaning that column so that its mean is zero). This is a destructive
     * operation that changes matrix {@code A} inplace.
     * 
     * @param A
     *            the matrix that contains the non-centered data
     * @return the matrix {@code A} de-meaned inplace
     */
    public static ComplexMatrixD centerInplace(ComplexMatrixD A) {
        double[] _a = A.getArrayUnsafe();
        int rows_ = A.numRows();
        int cols_ = A.numColumns();
        for (int col = 0; col < cols_; ++col) {
            double reColSum = 0.0;
            double imColSum = 0.0;
            double reColMean = 0.0;
            double imColMean = 0.0;
            for (int idx = 2 * col * rows_; idx < 2 * (col + 1) * rows_; idx += 2) {
                reColSum += _a[idx];
                imColSum += _a[idx + 1];
            }
            reColMean = reColSum / rows_;
            imColMean = imColSum / rows_;
            for (int idx = 2 * col * rows_; idx < 2 * (col + 1) * rows_; idx += 2) {
                _a[idx] -= reColMean;
                _a[idx + 1] -= imColMean;
            }
        }
        return A;
    }

    /**
     * Subtracts the mean of each column {@code j} from each value in that
     * column {@code j}, effectively centering each column around its mean
     * (de-meaning that column so that its mean is zero). This is a destructive
     * operation that changes matrix {@code A} inplace.
     * 
     * @param A
     *            the matrix that contains the non-centered data
     * @return the matrix {@code A} de-meaned inplace
     */
    public static ComplexMatrixF centerInplace(ComplexMatrixF A) {
        float[] _a = A.getArrayUnsafe();
        int rows_ = A.numRows();
        int cols_ = A.numColumns();
        for (int col = 0; col < cols_; ++col) {
            float reColSum = 0.0f;
            float imColSum = 0.0f;
            float reColMean = 0.0f;
            float imColMean = 0.0f;
            for (int idx = 2 * col * rows_; idx < 2 * (col + 1) * rows_; idx += 2) {
                reColSum += _a[idx];
                imColSum += _a[idx + 1];
            }
            reColMean = reColSum / rows_;
            imColMean = imColSum / rows_;
            for (int idx = 2 * col * rows_; idx < 2 * (col + 1) * rows_; idx += 2) {
                _a[idx] -= reColMean;
                _a[idx + 1] -= imColMean;
            }
        }
        return A;
    }

    private Statistics() {
        throw new AssertionError();
    }
}
