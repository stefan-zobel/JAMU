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

/**
 * Approximately optimal Singular Value truncation ("Singular Values Hard
 * Threshold (SVHT)") after Gavish and Donoho (2014).
 * <p>
 * Every method here expects the singular values in descending order, as LAPACK
 * hands them out, and takes {@code singularValues[0]} to be the largest of
 * them. {@code TOL_DBL} and {@code TOL_FLT} are relative to that largest value,
 * not absolute: a singular value carries the scale of the matrix it came from,
 * so an absolute machine epsilon compared against one makes the result depend
 * on how the input happened to be scaled. Measured before this was relative, one
 * and the same rank 5 spectrum answered 5 down to a scale of {@code 1e-13},
 * then 1, and 0 from {@code 1e-20} on; in single precision the same collapse
 * began at {@code 1e-6}.
 *
 * @see "https://arxiv.org/pdf/1305.5870.pdf"
 */
class SVHT {

    /** relative to the largest singular value, see the class comment */
    static final double TOL_DBL = 5.0 * DimensionsBase.MACH_EPS_DBL;
    /** relative to the largest singular value, see the class comment */
    static final float TOL_FLT = 5.0f * DimensionsBase.MACH_EPS_FLT;
    static final double BROAD_SHARE_DBL = 1.0 - 1e-4;
    static final float BROAD_SHARE_FLT = 1.0f - 1e-4f;

    static int threshold(int rows, int cols, double[] singularValues) {
        DimensionsBase.checkRows(rows);
        DimensionsBase.checkCols(cols);
        // there is nothing to threshold without a positive largest singular
        // value, and the negated comparison rejects a NaN as well
        if (!(singularValues[0] > 0.0)) {
            return 0;
        }
        double omega = computeOmega(rows, cols);
        double median = median(singularValues);
        double cutoff = omega * median;
        return threshold_(singularValues, cutoff);
    }

    static int threshold(int rows, int cols, float[] singularValues) {
        DimensionsBase.checkRows(rows);
        DimensionsBase.checkCols(cols);
        // there is nothing to threshold without a positive largest singular
        // value, and the negated comparison rejects a NaN as well
        if (!(singularValues[0] > 0.0f)) {
            return 0;
        }
        float omega = (float) computeOmega(rows, cols);
        float median = median(singularValues);
        float cutoff = omega * median;
        return threshold_(singularValues, cutoff);
    }

    static double getSigmaMin(double[] singularValues) {
        double tol = TOL_DBL * singularValues[0];
        for (int i = singularValues.length - 1; i >= 0; --i) {
            if (singularValues[i] > tol) {
                return singularValues[i];
            }
        }
        return singularValues[0];
    }

    static float getSigmaMin(float[] singularValues) {
        float tol = TOL_FLT * singularValues[0];
        for (int i = singularValues.length - 1; i >= 0; --i) {
            if (singularValues[i] > tol) {
                return singularValues[i];
            }
        }
        return singularValues[0];
    }

    static double median(double[] values) {
        // relative to the largest value, which is values[0] for a descending
        // spectrum. That also makes the fall through below unreachable: the
        // loop breaks at i = 0 at the latest, because values[0] is always
        // above 5 * eps * values[0] for a positive values[0]
        double tol = TOL_DBL * values[0];
        int len = values.length;
        int endIdx = len - 1;
        for (int i = endIdx; i >= 0; --i) {
            if (values[i] > tol) {
                endIdx = i;
                break;
            }
        }
        if (endIdx < len - 1) {
            len = endIdx + 1;
        }
        if (len % 2 != 0) {
            return values[(len - 1) / 2];
        } else {
            int mid = len / 2;
            return (values[mid - 1] + values[mid]) / 2.0;
        }
    }

    static float median(float[] values) {
        // relative to the largest value, see the double variant above
        float tol = TOL_FLT * values[0];
        int len = values.length;
        int endIdx = len - 1;
        for (int i = endIdx; i >= 0; --i) {
            if (values[i] > tol) {
                endIdx = i;
                break;
            }
        }
        if (endIdx < len - 1) {
            len = endIdx + 1;
        }
        if (len % 2 != 0) {
            return values[(len - 1) / 2];
        } else {
            int mid = len / 2;
            return (values[mid - 1] + values[mid]) / 2.0f;
        }
    }

    static double computeOmega(int rows, int cols) {
        int m = Math.min(rows, cols);
        int n = Math.max(rows, cols);
        double beta = m / (double) n;
        double betaSqr = beta * beta;
        double betaCub = betaSqr * beta;
        return 0.56 * betaCub - 0.95 * betaSqr + 1.82 * beta + 1.43;
    }

    private static int threshold_(double[] singularValues, double cutoff) {
        if (singularValues[0] < cutoff) {
            return 0;
        }
        int idx = 0;
        for (int i = 0; i < singularValues.length; ++i) {
            if (singularValues[i] <= cutoff) {
                // idx of last sv > cutoff
                idx = i - 1;
                break;
            }
        }
        if (idx > 0) {
            double cap = BROAD_SHARE_DBL * sum(singularValues);
            double sum = 0.0;
            int lastIdx = 0;
            for (int i = 0; i <= idx && sum < cap; ++i) {
                sum += singularValues[i];
                lastIdx = i;
            }
            idx = Math.min(idx, lastIdx);
        }
        // estimated optimal hard threshold
        idx = (idx < 0) ? 0 : idx;
        return idx + 1;
    }

    private static int threshold_(float[] singularValues, float cutoff) {
        if (singularValues[0] < cutoff) {
            return 0;
        }
        int idx = 0;
        for (int i = 0; i < singularValues.length; ++i) {
            if (singularValues[i] <= cutoff) {
                // idx of last sv > cutoff
                idx = i - 1;
                break;
            }
        }
        if (idx > 0) {
            float cap = BROAD_SHARE_FLT * sum(singularValues);
            float sum = 0.0f;
            int lastIdx = 0;
            for (int i = 0; i <= idx && sum < cap; ++i) {
                sum += singularValues[i];
                lastIdx = i;
            }
            idx = Math.min(idx, lastIdx);
        }
        // estimated optimal hard threshold
        idx = (idx < 0) ? 0 : idx;
        return idx + 1;
    }

    static double sum(double[] values) {
        // relative to the largest value, which also keeps the sum from coming
        // out as zero for a matrix of small scale. It used to, and the caller
        // then capped its answer at 1 whatever the spectrum said
        double tol = TOL_DBL * values[0];
        double sum = 0.0;
        for (int i = 0; i < values.length; ++i) {
            double sv = values[i];
            if (sv <= tol) {
                break;
            }
            sum += sv;
        }
        return sum;
    }

    static float sum(float[] values) {
        // relative to the largest value, see the double variant above
        float tol = TOL_FLT * values[0];
        float sum = 0.0f;
        for (int i = 0; i < values.length; ++i) {
            float sv = values[i];
            if (sv <= tol) {
                break;
            }
            sum += sv;
        }
        return sum;
    }

    private SVHT() {
        throw new AssertionError();
    }
}
