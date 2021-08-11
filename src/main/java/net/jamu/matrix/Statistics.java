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
 * Some static utility methods for matrices that may be useful in statistical
 * applications.
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

    public static MatrixD zscore(MatrixD A) {
        return zscoreInplace(A.copy());
    }

    public static MatrixF zscore(MatrixF A) {
        return zscoreInplace(A.copy());
    }

    public static ComplexMatrixD zscore(ComplexMatrixD A) {
        return zscoreInplace(A.copy());
    }

    public static ComplexMatrixF zscore(ComplexMatrixF A) {
        return zscoreInplace(A.copy());
    }

    public static MatrixD zscoreInplace(MatrixD A) {
        int rows_ = checkNotRowVector(A);
        int cols_ = A.numColumns();
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
                if (xi != 0.0) {
                    mean = (((count - 1) * mean) + xi) / count;
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
            double stddev = patchDev(scale * Math.sqrt(sumsquared / rows_ - y * y));
            for (int i = col * rows_; i < (col + 1) * rows_; ++i) {
                // subtract mean and divide by standard deviation
                double xi = _a[i];
                xi = (xi - mean) / stddev;
                _a[i] = xi;
            }
        }
        return A;
    }

    public static MatrixF zscoreInplace(MatrixF A) {
        int rows_ = checkNotRowVector(A);
        int cols_ = A.numColumns();
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
                if (xi != 0.0f) {
                    mean = (((count - 1) * mean) + xi) / count;
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
            float stddev = patchDev(scale * (float) Math.sqrt(sumsquared / rows_ - y * y));
            for (int i = col * rows_; i < (col + 1) * rows_; ++i) {
                // subtract mean and divide by standard deviation
                float xi = _a[i];
                xi = (xi - mean) / stddev;
                _a[i] = xi;
            }
        }
        return A;
    }

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
                if (xre != 0.0 || xim != 0.0) {
                    reMean = (((count - 1) * reMean) + xre) / count;
                    imMean = (((count - 1) * imMean) + xim) / count;
                    double absxre = Math.abs(xre);
                    double absxim = Math.abs(xim);
                    if (reScale < absxre) {
                        double unsquared = reScale / absxre;
                        reSumSqr = 1.0 + reSumSqr * (unsquared * unsquared);
                        reScale = absxre;
                    } else {
                        double unsquared = absxre / reScale;
                        reSumSqr = reSumSqr + (unsquared * unsquared);
                    }
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
            double reStddev = patchDev(reScale * Math.sqrt(reSumSqr / rows_ - reY * reY));
            double imStddev = patchDev(imScale * Math.sqrt(imSumSqr / rows_ - imY * imY));
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
                if (xre != 0.0f || xim != 0.0f) {
                    reMean = (((count - 1) * reMean) + xre) / count;
                    imMean = (((count - 1) * imMean) + xim) / count;
                    float absxre = Math.abs(xre);
                    float absxim = Math.abs(xim);
                    if (reScale < absxre) {
                        float unsquared = reScale / absxre;
                        reSumSqr = 1.0f + reSumSqr * (unsquared * unsquared);
                        reScale = absxre;
                    } else {
                        float unsquared = absxre / reScale;
                        reSumSqr = reSumSqr + (unsquared * unsquared);
                    }
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
            float reStddev = patchDev(reScale * (float) Math.sqrt(reSumSqr / rows_ - reY * reY));
            float imStddev = patchDev(imScale * (float) Math.sqrt(imSumSqr / rows_ - imY * imY));
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

    private static int checkNotRowVector(Dimensions A) {
        int rows = A.numRows();
        if (rows == 1) {
            throw new IllegalArgumentException("Can't compute zscore for a row vector");
        }
        return rows;
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
