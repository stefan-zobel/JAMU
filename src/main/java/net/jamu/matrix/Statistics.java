/*
 * Copyright 2021, 2024 Stefan Zobel
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
 * Some static utility methods for matrices that may be useful in statistical
 * applications.
 * 
 * @since 1.3
 */
public final class Statistics {

    /**
     * Holder for the first two moments of {@code MatrixD} columns or rows.
     * 
     * @since 1.4.6
     */
    public static class MomentsD {
        public MatrixD means;
        public MatrixD variances;

        /**
         * For use as an out parameter.
         */
        public MomentsD() {
        }

        public MomentsD(MatrixD means, MatrixD variances) {
            this.means = means;
            this.variances = variances;
        }

        @Override
        public String toString() {
            return new StringBuilder().append("means: ").append(means).append("variances: ").append(variances)
                    .toString();
        }
    }

    /**
     * Holder for the first two moments of {@code MatrixF} columns or rows.
     * 
     * @since 1.4.6
     */
    public static class MomentsF {
        public MatrixF means;
        public MatrixF variances;

        /**
         * For use as an out parameter.
         */
        public MomentsF() {
        }

        public MomentsF(MatrixF means, MatrixF variances) {
            this.means = means;
            this.variances = variances;
        }

        @Override
        public String toString() {
            return new StringBuilder().append("means: ").append(means).append("variances: ").append(variances)
                    .toString();
        }
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
            int count = 0;
            double colMean = 0.0;
            for (int idx = col * rows_; idx < (col + 1) * rows_; ++idx) {
                ++count;
                colMean = (((count - 1) * colMean) + _a[idx]) / count;
            }
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
            int count = 0;
            float colMean = 0.0f;
            for (int idx = col * rows_; idx < (col + 1) * rows_; ++idx) {
                ++count;
                colMean = (((count - 1) * colMean) + _a[idx]) / count;
            }
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
            int count = 0;
            double reColMean = 0.0;
            double imColMean = 0.0;
            for (int idx = 2 * col * rows_; idx < 2 * (col + 1) * rows_; idx += 2) {
                ++count;
                reColMean = (((count - 1) * reColMean) + _a[idx]) / count;
                imColMean = (((count - 1) * imColMean) + _a[idx + 1]) / count;
            }
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
            int count = 0;
            float reColMean = 0.0f;
            float imColMean = 0.0f;
            for (int idx = 2 * col * rows_; idx < 2 * (col + 1) * rows_; idx += 2) {
                ++count;
                reColMean = (((count - 1) * reColMean) + _a[idx]) / count;
                imColMean = (((count - 1) * imColMean) + _a[idx + 1]) / count;
            }
            for (int idx = 2 * col * rows_; idx < 2 * (col + 1) * rows_; idx += 2) {
                _a[idx] -= reColMean;
                _a[idx + 1] -= imColMean;
            }
        }
        return A;
    }

    /**
     * Creates a standard-scored copy of matrix {@code A}, i.e., the values in
     * each column of the copy have the mean of the corresponding column in
     * {@code A} subtracted and are then scaled to the unit variance of this
     * column by dividing the differences by the standard deviation of the
     * column values. The result is the signed number of standard deviations
     * (z-score) by which the value is above (below) the mean value of what is
     * being measured in the corresponding column. Matrix {@code A} doesn't get
     * mutated by this computation.
     * 
     * @param A
     *            the matrix whose columns contain the observations to be
     *            z-scored
     * @return a copy of A that has been z-scored
     */
    public static MatrixD zscore(MatrixD A) {
        return zscoreInplace(A.copy());
    }

    /**
     * Creates a standard-scored copy of matrix {@code A}, i.e., the values in
     * each column of the copy have the mean of the corresponding column in
     * {@code A} subtracted and are then scaled to the unit variance of this
     * column by dividing the differences by the standard deviation of the
     * column values. The result is the signed number of standard deviations
     * (z-score) by which the value is above (below) the mean value of what is
     * being measured in the corresponding column. Matrix {@code A} doesn't get
     * mutated by this computation.
     * 
     * @param A
     *            the matrix whose columns contain the observations to be
     *            z-scored
     * @return a copy of A that has been z-scored
     */
    public static MatrixF zscore(MatrixF A) {
        return zscoreInplace(A.copy());
    }

    /**
     * Creates a standard-scored copy of matrix {@code A}, i.e., the values in
     * each column of the copy have the mean of the corresponding column in
     * {@code A} subtracted and are then scaled to the unit variance of this
     * column by dividing the differences by the standard deviation of the
     * column values. The result is the signed number of standard deviations
     * (z-score) by which the value is above (below) the mean value of what is
     * being measured in the corresponding column. Matrix {@code A} doesn't get
     * mutated by this computation.
     * 
     * @param A
     *            the matrix whose columns contain the observations to be
     *            z-scored
     * @return a copy of A that has been z-scored
     */
    public static ComplexMatrixD zscore(ComplexMatrixD A) {
        return zscoreInplace(A.copy());
    }

    /**
     * Creates a standard-scored copy of matrix {@code A}, i.e., the values in
     * each column of the copy have the mean of the corresponding column in
     * {@code A} subtracted and are then scaled to the unit variance of this
     * column by dividing the differences by the standard deviation of the
     * column values. The result is the signed number of standard deviations
     * (z-score) by which the value is above (below) the mean value of what is
     * being measured in the corresponding column. Matrix {@code A} doesn't get
     * mutated by this computation.
     * 
     * @param A
     *            the matrix whose columns contain the observations to be
     *            z-scored
     * @return a copy of A that has been z-scored
     */
    public static ComplexMatrixF zscore(ComplexMatrixF A) {
        return zscoreInplace(A.copy());
    }

    /**
     * Subtracts the mean of each column {@code j} from each value in that
     * column {@code j} and then divides the difference by the standard
     * deviation of the values in column {@code j}, effectively expressing the
     * values in each column as the signed number of standard deviations
     * (z-score) by which they are above or below the column's mean value. This
     * is a destructive operation that changes matrix {@code A} inplace.
     * 
     * @param A
     *            the matrix whose columns contain the observations to be
     *            z-scored
     * @return the matrix {@code A} z-scored inplace
     */
    public static MatrixD zscoreInplace(MatrixD A) {
        return zscoreInplace(A, null);
    }

    /**
     * Subtracts the mean of each column {@code j} from each value in that
     * column {@code j} and then divides the difference by the standard
     * deviation of the values in column {@code j}, effectively expressing the
     * values in each column as the signed number of standard deviations
     * (z-score) by which they are above or below the column's mean value.
     * Optionally fills in the first two moments for each column as a row vector
     * in the {@code moments} argument if that is not {@code null}. This is a
     * destructive operation that changes matrix {@code A} inplace.
     * 
     * @param A
     *            the matrix whose columns contain the observations to be
     *            z-scored
     * @param moments
     *            optional holder object for the first two moments of each
     *            column, may be {@code null}
     * @return the matrix {@code A} z-scored inplace
     * @since 1.4.6
     */
    public static MatrixD zscoreInplace(MatrixD A, MomentsD moments) {
        int rows_ = checkNotRowVector(A);
        int cols_ = A.numColumns();
        if (moments != null) {
            if (moments.means == null || !(moments.means.isRowVector() && moments.means.numColumns() == cols_)) {
                moments.means = Matrices.createD(1, cols_);
            }
            if (moments.variances == null
                    || !(moments.variances.isRowVector() && moments.variances.numColumns() == cols_)) {
                moments.variances = Matrices.createD(1, cols_);
            }
        }
        double[] _a = A.getArrayUnsafe();
        for (int col = 0; col < cols_; ++col) {
            // overflow resistant implementation
            int count = 0;
            double mean = 0.0;
            double scale = 0.0;
            double sumsquared = 1.0;
            // determine mean and sum squared
            for (int i = col * rows_; i < (col + 1) * rows_; ++i) {
                ++count;
                double xi = _a[i];
                mean = (((count - 1) * mean) + xi) / count;
                if (xi != 0.0) {
                    double absxi = Math.abs(xi);
                    if (scale < absxi) {
                        double unsquared = scale / absxi;
                        sumsquared = 1.0 + sumsquared * (unsquared * unsquared);
                        scale = absxi;
                    } else {
                        double unsquared = absxi / scale;
                        sumsquared = sumsquared + (unsquared * unsquared);
                    }
                }
            }
            double y = computeScaledMean(scale, mean);
            double oneOverSqrtRows = 1.0 / Math.sqrt(rows_);
            double stddev = patchDev(scale * oneOverSqrtRows * Math.sqrt(sumsquared / rows_ - y * y));
            for (int i = col * rows_; i < (col + 1) * rows_; ++i) {
                // subtract mean and divide by standard deviation
                double xi = _a[i];
                xi = (xi - mean) / stddev;
                _a[i] = xi;
            }
            if (moments != null) {
                moments.means.setUnsafe(0, col, mean);
                moments.variances.setUnsafe(0, col, stddev * stddev);
            }
        }
        return A;
    }

    /**
     * Subtracts the mean of each column {@code j} from each value in that
     * column {@code j} and then divides the difference by the standard
     * deviation of the values in column {@code j}, effectively expressing the
     * values in each column as the signed number of standard deviations
     * (z-score) by which they are above or below the column's mean value. This
     * is a destructive operation that changes matrix {@code A} inplace.
     * 
     * @param A
     *            the matrix whose columns contain the observations to be
     *            z-scored
     * @return the matrix {@code A} z-scored inplace
     */
    public static MatrixF zscoreInplace(MatrixF A) {
        return zscoreInplace(A, null);
    }

    /**
     * Subtracts the mean of each column {@code j} from each value in that
     * column {@code j} and then divides the difference by the standard
     * deviation of the values in column {@code j}, effectively expressing the
     * values in each column as the signed number of standard deviations
     * (z-score) by which they are above or below the column's mean value.
     * Optionally fills in the first two moments for each column as a row vector
     * in the {@code moments} argument if that is not {@code null}. This is a
     * destructive operation that changes matrix {@code A} inplace.
     * 
     * @param A
     *            the matrix whose columns contain the observations to be
     *            z-scored
     * @param moments
     *            optional holder object for the first two moments of each
     *            column, may be {@code null}
     * @return the matrix {@code A} z-scored inplace
     * @since 1.4.6
     */
    public static MatrixF zscoreInplace(MatrixF A, MomentsF moments) {
        int rows_ = checkNotRowVector(A);
        int cols_ = A.numColumns();
        if (moments != null) {
            if (moments.means == null || !(moments.means.isRowVector() && moments.means.numColumns() == cols_)) {
                moments.means = Matrices.createF(1, cols_);
            }
            if (moments.variances == null
                    || !(moments.variances.isRowVector() && moments.variances.numColumns() == cols_)) {
                moments.variances = Matrices.createF(1, cols_);
            }
        }
        float[] _a = A.getArrayUnsafe();
        for (int col = 0; col < cols_; ++col) {
            // overflow resistant implementation
            int count = 0;
            float mean = 0.0f;
            float scale = 0.0f;
            float sumsquared = 1.0f;
            // determine mean and sum squared
            for (int i = col * rows_; i < (col + 1) * rows_; ++i) {
                ++count;
                float xi = _a[i];
                mean = (((count - 1) * mean) + xi) / count;
                if (xi != 0.0f) {
                    float absxi = Math.abs(xi);
                    if (scale < absxi) {
                        float unsquared = scale / absxi;
                        sumsquared = 1.0f + sumsquared * (unsquared * unsquared);
                        scale = absxi;
                    } else {
                        float unsquared = absxi / scale;
                        sumsquared = sumsquared + (unsquared * unsquared);
                    }
                }
            }
            float y = computeScaledMean(scale, mean);
            float oneOverSqrtRows = (float) (1.0 / Math.sqrt(rows_));
            float stddev = patchDev(scale * oneOverSqrtRows * (float) Math.sqrt(sumsquared / rows_ - y * y));
            for (int i = col * rows_; i < (col + 1) * rows_; ++i) {
                // subtract mean and divide by standard deviation
                float xi = _a[i];
                xi = (xi - mean) / stddev;
                _a[i] = xi;
            }
            if (moments != null) {
                moments.means.setUnsafe(0, col, mean);
                moments.variances.setUnsafe(0, col, stddev * stddev);
            }
        }
        return A;
    }

    /**
     * Subtracts the mean of each row {@code i} from each value in that row
     * {@code i} and then divides the difference by the standard deviation of
     * the values in row {@code i}, effectively expressing the values in each
     * row as the signed number of standard deviations (z-score) by which they
     * are above or below the row's mean value. Optionally fills in the first
     * two moments for each row as a column vector in the {@code moments}
     * argument if that is not {@code null}. This is a destructive operation
     * that changes matrix {@code A} inplace.
     * 
     * @param A
     *            the matrix whose rows contain the observations to be z-scored
     * @param moments
     *            optional holder object for the first two moments of each row,
     *            may be {@code null}
     * @return the matrix {@code A} z-scored inplace
     * @since 1.4.6
     */
    public static MatrixD zscoreRowsInplace(MatrixD A, MomentsD moments) {
        int cols_ = checkNotColumnVector(A);
        int rows_ = A.numRows();
        if (moments != null) {
            if (moments.means == null || !(moments.means.isColumnVector() && moments.means.numRows() == rows_)) {
                moments.means = Matrices.createD(rows_, 1);
            }
            if (moments.variances == null
                    || !(moments.variances.isColumnVector() && moments.variances.numRows() == rows_)) {
                moments.variances = Matrices.createD(rows_, 1);
            }
        }
        double[] _a = A.getArrayUnsafe();
        for (int row = 0; row < rows_; ++row) {
            // overflow resistant implementation
            int count = 0;
            double mean = 0.0;
            double scale = 0.0;
            double sumsquared = 1.0;
            // determine mean and sum squared
            for (int i = row; i < row + rows_ * cols_; i += rows_) {
                ++count;
                double xi = _a[i];
                mean = (((count - 1) * mean) + xi) / count;
                if (xi != 0.0) {
                    double absxi = Math.abs(xi);
                    if (scale < absxi) {
                        double unsquared = scale / absxi;
                        sumsquared = 1.0 + sumsquared * (unsquared * unsquared);
                        scale = absxi;
                    } else {
                        double unsquared = absxi / scale;
                        sumsquared = sumsquared + (unsquared * unsquared);
                    }
                }
            }
            double y = computeScaledMean(scale, mean);
            double oneOverSqrtCols = 1.0 / Math.sqrt(cols_);
            double stddev = patchDev(scale * oneOverSqrtCols * Math.sqrt(sumsquared / cols_ - y * y));
            for (int i = row; i < row + rows_ * cols_; i += rows_) {
                // subtract mean and divide by standard deviation
                double xi = _a[i];
                xi = (xi - mean) / stddev;
                _a[i] = xi;
            }
            if (moments != null) {
                moments.means.setUnsafe(row, 0, mean);
                moments.variances.setUnsafe(row, 0, stddev * stddev);
            }
        }
        return A;
    }

    /**
     * Subtracts the mean of each row {@code i} from each value in that row
     * {@code i} and then divides the difference by the standard deviation of
     * the values in row {@code i}, effectively expressing the values in each
     * row as the signed number of standard deviations (z-score) by which they
     * are above or below the row's mean value. Optionally fills in the first
     * two moments for each row as a column vector in the {@code moments}
     * argument if that is not {@code null}. This is a destructive operation
     * that changes matrix {@code A} inplace.
     * 
     * @param A
     *            the matrix whose rows contain the observations to be z-scored
     * @param moments
     *            optional holder object for the first two moments of each row,
     *            may be {@code null}
     * @return the matrix {@code A} z-scored inplace
     * @since 1.4.6
     */
    public static MatrixF zscoreRowsInplace(MatrixF A, MomentsF moments) {
        int cols_ = checkNotColumnVector(A);
        int rows_ = A.numRows();
        if (moments != null) {
            if (moments.means == null || !(moments.means.isColumnVector() && moments.means.numRows() == rows_)) {
                moments.means = Matrices.createF(rows_, 1);
            }
            if (moments.variances == null
                    || !(moments.variances.isColumnVector() && moments.variances.numRows() == rows_)) {
                moments.variances = Matrices.createF(rows_, 1);
            }
        }
        float[] _a = A.getArrayUnsafe();
        for (int row = 0; row < rows_; ++row) {
            // overflow resistant implementation
            int count = 0;
            float mean = 0.0f;
            float scale = 0.0f;
            float sumsquared = 1.0f;
            // determine mean and sum squared
            for (int i = row; i < row + rows_ * cols_; i += rows_) {
                ++count;
                float xi = _a[i];
                mean = (((count - 1) * mean) + xi) / count;
                if (xi != 0.0f) {
                    float absxi = Math.abs(xi);
                    if (scale < absxi) {
                        float unsquared = scale / absxi;
                        sumsquared = 1.0f + sumsquared * (unsquared * unsquared);
                        scale = absxi;
                    } else {
                        float unsquared = absxi / scale;
                        sumsquared = sumsquared + (unsquared * unsquared);
                    }
                }
            }
            float y = computeScaledMean(scale, mean);
            float oneOverSqrtCols = (float) (1.0 / Math.sqrt(cols_));
            float stddev = patchDev(scale * oneOverSqrtCols * (float) Math.sqrt(sumsquared / cols_ - y * y));
            for (int i = row; i < row + rows_ * cols_; i += rows_) {
                // subtract mean and divide by standard deviation
                float xi = _a[i];
                xi = (xi - mean) / stddev;
                _a[i] = xi;
            }
            if (moments != null) {
                moments.means.setUnsafe(row, 0, mean);
                moments.variances.setUnsafe(row, 0, stddev * stddev);
            }
        }
        return A;
    }

    /**
     * Rescales all elements in the matrix {@code A} into the range
     * {@code [lowerBound, upperBound]}.
     * 
     * @param A
     *            the matrix to rescale in-place
     * @param lowerBound
     *            the minimum value of an element after rescaling
     * @param upperBound
     *            the maximum value of an element after rescaling
     * @return the matrix {@code A} with all elements rescaled in-place
     * @since 1.4.6
     */
    public static MatrixD rescaleInplace(MatrixD A, double lowerBound, double upperBound) {
        double[] _a = A.getArrayUnsafe();
        double _min = Double.MAX_VALUE;
        double _max = -Double.MAX_VALUE;
        for (int i = 0; i < _a.length; ++i) {
            double x = _a[i];
            if (x < _min) {
                _min = x;
            }
            if (x > _max) {
                _max = x;
            }
        }
        double scale = upperBound - lowerBound;
        double dataScale = (_min == _max) ? Double.MIN_NORMAL : (_max - _min);
        for (int i = 0; i < _a.length; ++i) {
            _a[i] = lowerBound + (((_a[i] - _min) * scale) / dataScale);
        }
        return A;
    }

    /**
     * Rescales all elements in the matrix {@code A} into the range
     * {@code [lowerBound, upperBound]}.
     * 
     * @param A
     *            the matrix to rescale in-place
     * @param lowerBound
     *            the minimum value of an element after rescaling
     * @param upperBound
     *            the maximum value of an element after rescaling
     * @return the matrix {@code A} with all elements rescaled in-place
     * @since 1.4.6
     */
    public static MatrixF rescaleInplace(MatrixF A, float lowerBound, float upperBound) {
        float[] _a = A.getArrayUnsafe();
        float _min = Float.MAX_VALUE;
        float _max = -Float.MAX_VALUE;
        for (int i = 0; i < _a.length; ++i) {
            float x = _a[i];
            if (x < _min) {
                _min = x;
            }
            if (x > _max) {
                _max = x;
            }
        }
        float scale = upperBound - lowerBound;
        float dataScale = (_min == _max) ? Float.MIN_NORMAL : (_max - _min);
        for (int i = 0; i < _a.length; ++i) {
            _a[i] = lowerBound + (((_a[i] - _min) * scale) / dataScale);
        }
        return A;
    }

    /**
     * Subtracts the mean of each column {@code j} from each value in that
     * column {@code j} and then divides the difference by the standard
     * deviation of the values in column {@code j}, effectively expressing the
     * values in each column as the signed number of standard deviations
     * (z-score) by which they are above or below the column's mean value. This
     * is a destructive operation that changes matrix {@code A} inplace.
     * 
     * @param A
     *            the matrix whose columns contain the observations to be
     *            z-scored
     * @return the matrix {@code A} z-scored inplace
     */
    public static ComplexMatrixD zscoreInplace(ComplexMatrixD A) {
        int rows_ = checkNotRowVector(A);
        int cols_ = A.numColumns();
        double[] _a = A.getArrayUnsafe();
        for (int col = 0; col < cols_; ++col) {
            // overflow resistant implementation
            int count = 0;
            double reMean = 0.0;
            double imMean = 0.0;
            double reScale = 0.0;
            double imScale = 0.0;
            double reSumSqr = 1.0;
            double imSumSqr = 1.0;
            // determine mean and sum squared
            for (int i = 2 * col * rows_; i < 2 * (col + 1) * rows_; i += 2) {
                ++count;
                double xre = _a[i];
                double xim = _a[i + 1];
                reMean = (((count - 1) * reMean) + xre) / count;
                imMean = (((count - 1) * imMean) + xim) / count;
                if (xre != 0.0) {
                    double absxre = Math.abs(xre);
                    if (reScale < absxre) {
                        double unsquared = reScale / absxre;
                        reSumSqr = 1.0 + reSumSqr * (unsquared * unsquared);
                        reScale = absxre;
                    } else {
                        double unsquared = absxre / reScale;
                        reSumSqr = reSumSqr + (unsquared * unsquared);
                    }
                }
                if (xim != 0.0) {
                    double absxim = Math.abs(xim);
                    if (imScale < absxim) {
                        double unsquared = imScale / absxim;
                        imSumSqr = 1.0 + imSumSqr * (unsquared * unsquared);
                        imScale = absxim;
                    } else {
                        double unsquared = absxim / imScale;
                        imSumSqr = imSumSqr + (unsquared * unsquared);
                    }
                }
            }
            //
            double reY = computeScaledMean(reScale, reMean);
            double imY = computeScaledMean(imScale, imMean);
            double oneOverSqrtRows = 1.0 / Math.sqrt(rows_);
            double reStddev = patchDev(reScale * oneOverSqrtRows * Math.sqrt(reSumSqr / rows_ - reY * reY));
            double imStddev = patchDev(imScale * oneOverSqrtRows * Math.sqrt(imSumSqr / rows_ - imY * imY));
            //
            for (int i = 2 * col * rows_; i < 2 * (col + 1) * rows_; i += 2) {
                // subtract mean and divide by standard deviation
                double xre = _a[i];
                double xim = _a[i + 1];
                xre = (xre - reMean) / reStddev;
                xim = (xim - imMean) / imStddev;
                _a[i] = xre;
                _a[i + 1] = xim;
            }
        }
        return A;
    }

    /**
     * Subtracts the mean of each column {@code j} from each value in that
     * column {@code j} and then divides the difference by the standard
     * deviation of the values in column {@code j}, effectively expressing the
     * values in each column as the signed number of standard deviations
     * (z-score) by which they are above or below the column's mean value. This
     * is a destructive operation that changes matrix {@code A} inplace.
     * 
     * @param A
     *            the matrix whose columns contain the observations to be
     *            z-scored
     * @return the matrix {@code A} z-scored inplace
     */
    public static ComplexMatrixF zscoreInplace(ComplexMatrixF A) {
        int rows_ = checkNotRowVector(A);
        int cols_ = A.numColumns();
        float[] _a = A.getArrayUnsafe();
        for (int col = 0; col < cols_; ++col) {
            // overflow resistant implementation
            int count = 0;
            float reMean = 0.0f;
            float imMean = 0.0f;
            float reScale = 0.0f;
            float imScale = 0.0f;
            float reSumSqr = 1.0f;
            float imSumSqr = 1.0f;
            // determine mean and sum squared
            for (int i = 2 * col * rows_; i < 2 * (col + 1) * rows_; i += 2) {
                ++count;
                float xre = _a[i];
                float xim = _a[i + 1];
                reMean = (((count - 1) * reMean) + xre) / count;
                imMean = (((count - 1) * imMean) + xim) / count;
                if (xre != 0.0f) {
                    float absxre = Math.abs(xre);
                    if (reScale < absxre) {
                        float unsquared = reScale / absxre;
                        reSumSqr = 1.0f + reSumSqr * (unsquared * unsquared);
                        reScale = absxre;
                    } else {
                        float unsquared = absxre / reScale;
                        reSumSqr = reSumSqr + (unsquared * unsquared);
                    }
                }
                if (xim != 0.0f) {
                    float absxim = Math.abs(xim);
                    if (imScale < absxim) {
                        float unsquared = imScale / absxim;
                        imSumSqr = 1.0f + imSumSqr * (unsquared * unsquared);
                        imScale = absxim;
                    } else {
                        float unsquared = absxim / imScale;
                        imSumSqr = imSumSqr + (unsquared * unsquared);
                    }
                }
            }
            //
            float reY = computeScaledMean(reScale, reMean);
            float imY = computeScaledMean(imScale, imMean);
            float oneOverSqrtRows = (float) (1.0 / Math.sqrt(rows_));
            float reStddev = patchDev(reScale * oneOverSqrtRows * (float) Math.sqrt(reSumSqr / rows_ - reY * reY));
            float imStddev = patchDev(imScale * oneOverSqrtRows * (float) Math.sqrt(imSumSqr / rows_ - imY * imY));
            //
            for (int i = 2 * col * rows_; i < 2 * (col + 1) * rows_; i += 2) {
                // subtract mean and divide by standard deviation
                float xre = _a[i];
                float xim = _a[i + 1];
                xre = (xre - reMean) / reStddev;
                xim = (xim - imMean) / imStddev;
                _a[i] = xre;
                _a[i + 1] = xim;
            }
        }
        return A;
    }

    /**
     * Randomly permutes the columns in matrix {@code A} in place using a
     * default source of randomness. All permutations occur with approximately
     * equal probability.
     * 
     * @param A
     *            the matrix whose columns will be permuted at random
     * @return the matrix argument with columns randomly permuted in place
     * @since 1.4.4
     */
    public static MatrixD shuffleColumnsInplace(MatrixD A) {
        return shuffleColumnsInplace(A, null);
    }

    /**
     * Randomly permutes the columns in matrix {@code A} in place using a
     * default source of randomness. All permutations occur with approximately
     * equal probability.
     * 
     * @param A
     *            the matrix whose columns will be permuted at random
     * @return the matrix argument with columns randomly permuted in place
     * @since 1.4.4
     */
    public static MatrixF shuffleColumnsInplace(MatrixF A) {
        return shuffleColumnsInplace(A, null);
    }

    /**
     * Randomly permutes the columns in matrix {@code A} in place using a
     * default source of randomness seeded by the given {@code seed}. All
     * permutations occur with approximately equal probability.
     * 
     * @param A
     *            the matrix whose columns will be permuted at random
     * @param seed
     *            the initial seed to use for the PRNG
     * @return the matrix argument with columns randomly permuted in place
     * @since 1.4.4
     */
    public static MatrixD shuffleColumnsInplace(MatrixD A, long seed) {
        return shuffleColumnsInplace(A, new XoShiRo256StarStar(seed));
    }

    /**
     * Randomly permutes the columns in matrix {@code A} in place using a
     * default source of randomness seeded by the given {@code seed}. All
     * permutations occur with approximately equal probability.
     * 
     * @param A
     *            the matrix whose columns will be permuted at random
     * @param seed
     *            the initial seed to use for the PRNG
     * @return the matrix argument with columns randomly permuted in place
     * @since 1.4.4
     */
    public static MatrixF shuffleColumnsInplace(MatrixF A, long seed) {
        return shuffleColumnsInplace(A, new XoShiRo256StarStar(seed));
    }

    private static MatrixD shuffleColumnsInplace(MatrixD A, XoShiRo256StarStar rng) {
        int rows = A.numRows();
        int cols = A.numColumns();
        double[] a = A.getArrayUnsafe();
        double[] tmp = new double[rows];
        XoShiRo256StarStar rnd = (rng == null) ? new XoShiRo256StarStar() : rng;
        for (int i = cols; i > 1; --i) {
            swap((i - 1) * rows, a, tmp, rows, rnd.nextInt(i) * rows);
        }
        return A;
    }

    private static MatrixF shuffleColumnsInplace(MatrixF A, XoShiRo256StarStar rng) {
        int rows = A.numRows();
        int cols = A.numColumns();
        float[] a = A.getArrayUnsafe();
        float[] tmp = new float[rows];
        XoShiRo256StarStar rnd = (rng == null) ? new XoShiRo256StarStar() : rng;
        for (int i = cols; i > 1; --i) {
            swap((i - 1) * rows, a, tmp, rows, rnd.nextInt(i) * rows);
        }
        return A;
    }

    private static void swap(int aoff1, Object a, Object tmp, int len, int aoff2) {
        if (aoff1 != aoff2) {
            System.arraycopy(a, aoff1, tmp, 0, len);
            System.arraycopy(a, aoff2, a, aoff1, len);
            System.arraycopy(tmp, 0, a, aoff2, len);
        }
    }

    /**
     * Randomly permutes the rows in matrix {@code A} in place using a
     * default source of randomness. All permutations occur with approximately
     * equal probability.
     * 
     * @param A
     *            the matrix whose rows will be permuted at random
     * @return the matrix argument with rows randomly permuted in place
     * @since 1.4.6
     */
    public static MatrixD shuffleRowsInplace(MatrixD A) {
        return shuffleRowsInplace(A, null);
    }

    /**
     * Randomly permutes the rows in matrix {@code A} in place using a
     * default source of randomness. All permutations occur with approximately
     * equal probability.
     * 
     * @param A
     *            the matrix whose rows will be permuted at random
     * @return the matrix argument with rows randomly permuted in place
     * @since 1.4.6
     */
    public static MatrixF shuffleRowsInplace(MatrixF A) {
        return shuffleRowsInplace(A, null);
    }

    /**
     * Randomly permutes the rows in matrix {@code A} in place using a
     * default source of randomness seeded by the given {@code seed}. All
     * permutations occur with approximately equal probability.
     * 
     * @param A
     *            the matrix whose rows will be permuted at random
     * @param seed
     *            the initial seed to use for the PRNG
     * @return the matrix argument with rows randomly permuted in place
     * @since 1.4.6
     */
    public static MatrixD shuffleRowsInplace(MatrixD A, long seed) {
        return shuffleRowsInplace(A, new XoShiRo256StarStar(seed));
    }

    /**
     * Randomly permutes the rows in matrix {@code A} in place using a
     * default source of randomness seeded by the given {@code seed}. All
     * permutations occur with approximately equal probability.
     * 
     * @param A
     *            the matrix whose rows will be permuted at random
     * @param seed
     *            the initial seed to use for the PRNG
     * @return the matrix argument with rows randomly permuted in place
     * @since 1.4.6
     */
    public static MatrixF shuffleRowsInplace(MatrixF A, long seed) {
        return shuffleRowsInplace(A, new XoShiRo256StarStar(seed));
    }

    private static MatrixD shuffleRowsInplace(MatrixD A, XoShiRo256StarStar rng) {
        int rows = A.numRows();
        int cols = A.numColumns();
        double[] a = A.getArrayUnsafe();
        double[] tmp = new double[cols];
        XoShiRo256StarStar rnd = (rng == null) ? new XoShiRo256StarStar() : rng;
        for (int i = rows; i > 1; --i) {
            int sourceRow = rnd.nextInt(i);
            int targetRow = i - 1;
            if (sourceRow != targetRow) {
                swapRows(targetRow, a, tmp, cols, rows, sourceRow);
            }
        }
        return A;
    }

    private static MatrixF shuffleRowsInplace(MatrixF A, XoShiRo256StarStar rng) {
        int rows = A.numRows();
        int cols = A.numColumns();
        float[] a = A.getArrayUnsafe();
        float[] tmp = new float[cols];
        XoShiRo256StarStar rnd = (rng == null) ? new XoShiRo256StarStar() : rng;
        for (int i = rows; i > 1; --i) {
            int sourceRow = rnd.nextInt(i);
            int targetRow = i - 1;
            if (sourceRow != targetRow) {
                swapRows(targetRow, a, tmp, cols, rows, sourceRow);
            }
        }
        return A;
    }

    private static void swapRows(int aoff1, double[] a, double[] tmp, int len, int skip, int aoff2) {
        int j = 0;
        for (int i = aoff1; i < aoff1 + skip * len; aoff2 += skip, i += skip) {
            tmp[j] = a[i];
            a[i] = a[aoff2];
            a[aoff2] = tmp[j];
            ++j;
        }
    }

    private static void swapRows(int aoff1, float[] a, float[] tmp, int len, int skip, int aoff2) {
        int j = 0;
        for (int i = aoff1; i < aoff1 + skip * len; aoff2 += skip, i += skip) {
            tmp[j] = a[i];
            a[i] = a[aoff2];
            a[aoff2] = tmp[j];
            ++j;
        }
    }

    private static int checkNotRowVector(MatrixDimensions A) {
        int rows = A.numRows();
        if (rows == 1) {
            throw new IllegalArgumentException("Can't compute zscore for a row vector");
        }
        return rows;
    }

    private static int checkNotColumnVector(MatrixDimensions A) {
        int cols = A.numColumns();
        if (cols == 1) {
            throw new IllegalArgumentException("Can't compute zscore for a column vector");
        }
        return cols;
    }

    private static double patchDev(double stddev) {
        return (stddev == 0.0 || Double.isNaN(stddev)) ? 1.0 : stddev;
    }

    private static float patchDev(float stddev) {
        return (stddev == 0.0f || Float.isNaN(stddev)) ? 1.0f : stddev;
    }

    private static double computeScaledMean(double scale, double mean) {
        return (scale != 0.0) ? mean / scale : mean;
    }

    private static float computeScaledMean(float scale, float mean) {
        return (scale != 0.0f) ? mean / scale : mean;
    }

    private Statistics() {
        throw new AssertionError();
    }
}
