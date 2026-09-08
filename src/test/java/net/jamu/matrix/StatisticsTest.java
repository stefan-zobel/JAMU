/*
 * Copyright 2026 Stefan Zobel
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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Tests for Statistics. The z-score cases use data whose coefficient of
 * variation is of order one.
 */
public final class StatisticsTest {

    private static final double TOL_D = 1.0e-12;
    private static final float TOL_F = 1.0e-4f;

    // the lengths at which a z-scored column used to come out with variance n
    private static final int[] LENGTHS = { 2, 3, 5, 10, 50, 200 };

    @Test
    public void testTheZScoreOfAUnitVarianceColumnIsItself() {
        // (1, -1, 1, -1) has mean 0 and variance 1, so it is its own z-score
        MatrixD d = Matrices.createD(4, 2);
        MatrixF f = Matrices.createF(4, 2);
        for (int col = 0; col < 2; ++col) {
            for (int row = 0; row < 4; ++row) {
                double v = (row % 2 == 0) ? 1.0 : -1.0;
                d.set(row, col, v);
                f.set(row, col, (float) v);
            }
        }
        MatrixD zd = Statistics.zscoreColumns(d);
        MatrixF zf = Statistics.zscoreColumns(f);
        for (int col = 0; col < 2; ++col) {
            for (int row = 0; row < 4; ++row) {
                assertEquals("MatrixD at " + row + "," + col, d.get(row, col), zd.get(row, col), 0.0);
                assertEquals("MatrixF at " + row + "," + col, f.get(row, col), zf.get(row, col), 0.0f);
            }
        }
        MatrixD rd = Statistics.zscoreRows(d.transpose());
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 4; ++col) {
                assertEquals("zscoreRows at " + row + "," + col, d.transpose().get(row, col),
                        rd.get(row, col), 0.0);
            }
        }
    }

    @Test
    public void testZScoredColumnsHaveUnitVarianceAndZeroMean() {
        for (int n : LENGTHS) {
            String at = " at n = " + n;
            MatrixD d = Statistics.zscoreColumns(dataD(n, 3, 5L));
            for (int col = 0; col < 3; ++col) {
                double[] mv = moments(column(d, col));
                assertEquals("MatrixD mean" + at, 0.0, mv[0], TOL_D);
                assertEquals("MatrixD variance" + at, 1.0, mv[1], TOL_D);
            }
            MatrixF f = Statistics.zscoreColumns(dataF(n, 3, 5L));
            for (int col = 0; col < 3; ++col) {
                double[] mv = moments(column(f, col));
                assertEquals("MatrixF mean" + at, 0.0, mv[0], TOL_F);
                assertEquals("MatrixF variance" + at, 1.0, mv[1], TOL_F);
            }
        }
    }

    @Test
    public void testZScoredRowsHaveUnitVarianceAndZeroMean() {
        for (int n : LENGTHS) {
            String at = " at n = " + n;
            MatrixD d = Statistics.zscoreRows(dataD(3, n, 5L));
            for (int row = 0; row < 3; ++row) {
                double[] mv = moments(row(d, row));
                assertEquals("MatrixD mean" + at, 0.0, mv[0], TOL_D);
                assertEquals("MatrixD variance" + at, 1.0, mv[1], TOL_D);
            }
            MatrixF f = Statistics.zscoreRows(dataF(3, n, 5L));
            for (int row = 0; row < 3; ++row) {
                double[] mv = moments(row(f, row));
                assertEquals("MatrixF mean" + at, 0.0, mv[0], TOL_F);
                assertEquals("MatrixF variance" + at, 1.0, mv[1], TOL_F);
            }
        }
    }

    @Test
    public void testZScoredComplexColumnsHaveUnitVarianceAndZeroMean() {
        for (int n : LENGTHS) {
            String at = " at n = " + n;
            ComplexMatrixD d = Statistics.zscoreColumns(Matrices.randomUniformComplexD(n, 3, -1.0, 1.0, 5L));
            for (int col = 0; col < 3; ++col) {
                double[] re = moments(complexColumn(d, col, true));
                double[] im = moments(complexColumn(d, col, false));
                assertEquals("ComplexMatrixD real mean" + at, 0.0, re[0], TOL_D);
                assertEquals("ComplexMatrixD real variance" + at, 1.0, re[1], TOL_D);
                assertEquals("ComplexMatrixD imag mean" + at, 0.0, im[0], TOL_D);
                assertEquals("ComplexMatrixD imag variance" + at, 1.0, im[1], TOL_D);
            }
            ComplexMatrixF f = Statistics
                    .zscoreColumns(Matrices.randomUniformComplexF(n, 3, -1.0f, 1.0f, 5L));
            for (int col = 0; col < 3; ++col) {
                double[] re = moments(complexColumn(f, col, true));
                double[] im = moments(complexColumn(f, col, false));
                assertEquals("ComplexMatrixF real mean" + at, 0.0, re[0], TOL_F);
                assertEquals("ComplexMatrixF real variance" + at, 1.0, re[1], TOL_F);
                assertEquals("ComplexMatrixF imag mean" + at, 0.0, im[0], TOL_F);
                assertEquals("ComplexMatrixF imag variance" + at, 1.0, im[1], TOL_F);
            }
        }
    }

    @Test
    public void testTheReportedMomentsMatchTheTrueMoments() {
        for (int n : LENGTHS) {
            String at = " at n = " + n;
            MatrixD dc = dataD(n, 3, 9L);
            Statistics.MomentsD md = new Statistics.MomentsD();
            Statistics.zscoreColumnsInplace(dc.copy(), md);
            for (int col = 0; col < 3; ++col) {
                double[] mv = moments(column(dc, col));
                assertEquals("MomentsD column mean" + at, mv[0], md.means.get(0, col), TOL_D);
                assertEquals("MomentsD column variance" + at, mv[1], md.variances.get(0, col), TOL_D);
            }
            MatrixD dr = dataD(3, n, 9L);
            Statistics.MomentsD mdr = new Statistics.MomentsD();
            Statistics.zscoreRowsInplace(dr.copy(), mdr);
            for (int r = 0; r < 3; ++r) {
                double[] mv = moments(row(dr, r));
                assertEquals("MomentsD row mean" + at, mv[0], mdr.means.get(r, 0), TOL_D);
                assertEquals("MomentsD row variance" + at, mv[1], mdr.variances.get(r, 0), TOL_D);
            }
            MatrixF fc = dataF(n, 3, 9L);
            Statistics.MomentsF mf = new Statistics.MomentsF();
            Statistics.zscoreColumnsInplace(fc.copy(), mf);
            for (int col = 0; col < 3; ++col) {
                double[] mv = moments(column(fc, col));
                assertEquals("MomentsF column mean" + at, mv[0], mf.means.get(0, col), TOL_F);
                assertEquals("MomentsF column variance" + at, mv[1], mf.variances.get(0, col), TOL_F);
            }
            MatrixF fr = dataF(3, n, 9L);
            Statistics.MomentsF mfr = new Statistics.MomentsF();
            Statistics.zscoreRowsInplace(fr.copy(), mfr);
            for (int r = 0; r < 3; ++r) {
                double[] mv = moments(row(fr, r));
                assertEquals("MomentsF row mean" + at, mv[0], mfr.means.get(r, 0), TOL_F);
                assertEquals("MomentsF row variance" + at, mv[1], mfr.variances.get(r, 0), TOL_F);
            }
        }
    }

    @Test
    public void testAConstantColumnBecomesZero() {
        MatrixD d = Matrices.createD(5, 2);
        for (int col = 0; col < 2; ++col) {
            for (int row = 0; row < 5; ++row) {
                d.set(row, col, 7.5);
            }
        }
        Statistics.MomentsD m = new Statistics.MomentsD();
        MatrixD z = Statistics.zscoreColumnsInplace(d.copy(), m);
        for (int col = 0; col < 2; ++col) {
            for (int row = 0; row < 5; ++row) {
                assertEquals("constant column", 0.0, z.get(row, col), 0.0);
            }
            assertEquals("mean of a constant column", 7.5, m.means.get(0, col), 0.0);
            assertEquals("variance of a constant column", 0.0, m.variances.get(0, col), 0.0);
        }
    }

    @Test
    public void testCenterColumnsMakesTheColumnMeansZero() {
        for (int n : LENGTHS) {
            String at = " at n = " + n;
            MatrixD d = Statistics.centerColumns(dataD(n, 3, 11L));
            MatrixF f = Statistics.centerColumns(dataF(n, 3, 11L));
            ComplexMatrixD cd = Statistics
                    .centerColumns(Matrices.randomUniformComplexD(n, 3, -1.0, 1.0, 11L));
            for (int col = 0; col < 3; ++col) {
                assertEquals("MatrixD" + at, 0.0, moments(column(d, col))[0], TOL_D);
                assertEquals("MatrixF" + at, 0.0, moments(column(f, col))[0], TOL_F);
                assertEquals("ComplexMatrixD real" + at, 0.0, moments(complexColumn(cd, col, true))[0],
                        TOL_D);
                assertEquals("ComplexMatrixD imag" + at, 0.0, moments(complexColumn(cd, col, false))[0],
                        TOL_D);
            }
        }
    }

    @Test
    public void testRescaleHitsTheBounds() {
        MatrixD d = Statistics.rescale(dataD(10, 4, 13L), -3.0, 7.0);
        double lo = Double.MAX_VALUE;
        double hi = -Double.MAX_VALUE;
        for (double x : d.getArrayUnsafe()) {
            lo = Math.min(lo, x);
            hi = Math.max(hi, x);
        }
        assertEquals("MatrixD lower bound", -3.0, lo, TOL_D);
        assertEquals("MatrixD upper bound", 7.0, hi, TOL_D);
        MatrixF f = Statistics.rescale(dataF(10, 4, 13L), -3.0f, 7.0f);
        float flo = Float.MAX_VALUE;
        float fhi = -Float.MAX_VALUE;
        for (float x : f.getArrayUnsafe()) {
            flo = Math.min(flo, x);
            fhi = Math.max(fhi, x);
        }
        assertEquals("MatrixF lower bound", -3.0f, flo, TOL_F);
        assertEquals("MatrixF upper bound", 7.0f, fhi, TOL_F);
    }

    @Test
    public void testShufflingIsAPermutation() {
        MatrixD a = dataD(6, 5, 17L);
        assertTrue("shuffleColumns", sameColumns(a, Statistics.shuffleColumns(a)));
        MatrixD b = Statistics.shuffleRows(a);
        assertTrue("shuffleRows", sameColumns(a.transpose(), b.transpose()));
    }

    private static MatrixD dataD(int rows, int cols, long seed) {
        return Matrices.randomUniformD(rows, cols, -1.0, 1.0, seed);
    }

    private static MatrixF dataF(int rows, int cols, long seed) {
        return Matrices.randomUniformF(rows, cols, -1.0f, 1.0f, seed);
    }

    private static double[] column(MatrixD m, int col) {
        double[] x = new double[m.numRows()];
        for (int i = 0; i < x.length; ++i) {
            x[i] = m.get(i, col);
        }
        return x;
    }

    private static double[] column(MatrixF m, int col) {
        double[] x = new double[m.numRows()];
        for (int i = 0; i < x.length; ++i) {
            x[i] = m.get(i, col);
        }
        return x;
    }

    private static double[] row(MatrixD m, int r) {
        double[] x = new double[m.numColumns()];
        for (int i = 0; i < x.length; ++i) {
            x[i] = m.get(r, i);
        }
        return x;
    }

    private static double[] row(MatrixF m, int r) {
        double[] x = new double[m.numColumns()];
        for (int i = 0; i < x.length; ++i) {
            x[i] = m.get(r, i);
        }
        return x;
    }

    private static double[] complexColumn(ComplexMatrixD m, int col, boolean real) {
        double[] x = new double[m.numRows()];
        for (int i = 0; i < x.length; ++i) {
            x[i] = real ? m.get(i, col).re() : m.get(i, col).im();
        }
        return x;
    }

    private static double[] complexColumn(ComplexMatrixF m, int col, boolean real) {
        double[] x = new double[m.numRows()];
        for (int i = 0; i < x.length; ++i) {
            x[i] = real ? m.get(i, col).re() : m.get(i, col).im();
        }
        return x;
    }

    // mean and population variance, two-pass so that the reference is stable
    private static double[] moments(double[] x) {
        double mean = 0.0;
        for (int i = 0; i < x.length; ++i) {
            mean += x[i];
        }
        mean /= x.length;
        double s = 0.0;
        for (int i = 0; i < x.length; ++i) {
            double d = x[i] - mean;
            s += d * d;
        }
        return new double[] { mean, s / x.length };
    }

    private static boolean sameColumns(MatrixD a, MatrixD b) {
        int n = a.numColumns();
        if (b.numColumns() != n || b.numRows() != a.numRows()) {
            return false;
        }
        boolean[] used = new boolean[n];
        for (int j = 0; j < n; ++j) {
            boolean found = false;
            for (int k = 0; k < n && !found; ++k) {
                if (!used[k] && java.util.Arrays.equals(column(a, k), column(b, j))) {
                    used[k] = true;
                    found = true;
                }
            }
            if (!found) {
                return false;
            }
        }
        return true;
    }
}
