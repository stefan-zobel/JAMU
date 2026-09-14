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
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

import org.junit.Test;

/**
 * Pins the six broadcasted row vector operations of {@code MatrixD} and
 * {@code MatrixF}: {@code addBroadcastedRowVectorInplace} /
 * {@code plusBroadcastedRowVector}, {@code mulBroadcastedRowVectorInplace} /
 * {@code mulBroadcastedRowVector} and {@code divBroadcastedRowVectorInplace} /
 * {@code divBroadcastedRowVector}.
 * <p>
 * A row vector must act as if it had been stretched across all rows, a matrix
 * of equal dimension must act elementwise, and any other dimension must be
 * rejected. The expected values come from a scalar loop written out in this
 * class, in the same operand order as the implementation, so the comparison is
 * on raw bits. As an independent check the row broadcast must also equal the
 * transposed column broadcast of {@link BroadcastedVectorTest}'s methods.
 */
public final class BroadcastedRowVectorTest {

    private static final long SEED = 20260913L;

    /** the three operations, in the order the reference loop switches on */
    private static final char ADD = '+';
    private static final char MUL = '*';
    private static final char DIV = '/';

    private static final char[] OPS = { ADD, MUL, DIV };

    private static final int[][] SHAPES = { { 1, 1 }, { 1, 5 }, { 5, 1 }, { 4, 3 }, { 3, 4 }, { 40, 7 },
            { 7, 40 } };

    // ---------------------------------------------------------------- tests

    @Test
    public void testARowVectorIsStretchedAcrossAllRows() {
        for (int[] s : SHAPES) {
            String at = " at " + s[0] + "x" + s[1];
            for (char op : OPS) {
                MatrixD d = Matrices.randomUniformD(s[0], s[1], 1.0, 2.0, SEED);
                MatrixD vd = Matrices.randomUniformD(1, s[1], 1.0, 2.0, SEED + 1L);
                double[] wantD = refD(d.getArrayUnsafe(), vd.getArrayUnsafe(), s[0], s[1], true, op);
                assertSame(op + " MatrixD returns this" + at, d, applyD(d, vd, op));
                assertBitsD(op + " MatrixD" + at, wantD, d.getArrayUnsafe());

                MatrixF f = Matrices.randomUniformF(s[0], s[1], 1.0f, 2.0f, SEED);
                MatrixF vf = Matrices.randomUniformF(1, s[1], 1.0f, 2.0f, SEED + 1L);
                float[] wantF = refF(f.getArrayUnsafe(), vf.getArrayUnsafe(), s[0], s[1], true, op);
                assertSame(op + " MatrixF returns this" + at, f, applyF(f, vf, op));
                assertBitsF(op + " MatrixF" + at, wantF, f.getArrayUnsafe());
            }
        }
    }

    @Test
    public void testAMatrixOfEqualDimensionActsElementwise() {
        for (int[] s : SHAPES) {
            String at = " at " + s[0] + "x" + s[1];
            for (char op : OPS) {
                MatrixD d = Matrices.randomUniformD(s[0], s[1], 1.0, 2.0, SEED);
                MatrixD bd = Matrices.randomUniformD(s[0], s[1], 1.0, 2.0, SEED + 1L);
                double[] wantD = refD(d.getArrayUnsafe(), bd.getArrayUnsafe(), s[0], s[1], false, op);
                applyD(d, bd, op);
                assertBitsD(op + " MatrixD" + at, wantD, d.getArrayUnsafe());

                MatrixF f = Matrices.randomUniformF(s[0], s[1], 1.0f, 2.0f, SEED);
                MatrixF bf = Matrices.randomUniformF(s[0], s[1], 1.0f, 2.0f, SEED + 1L);
                float[] wantF = refF(f.getArrayUnsafe(), bf.getArrayUnsafe(), s[0], s[1], false, op);
                applyF(f, bf, op);
                assertBitsF(op + " MatrixF" + at, wantF, f.getArrayUnsafe());
            }
        }
    }

    @Test
    public void testTheOneRowCaseAgreesWithTheEqualDimensionCase() {
        // a 1 row matrix hits the equal dimension branch, a taller one the
        // broadcast branch; stretching by hand must give the same answer
        for (int[] s : new int[][] { { 4, 6 }, { 3, 40 }, { 40, 3 } }) {
            for (char op : OPS) {
                MatrixD d = Matrices.randomUniformD(s[0], s[1], 1.0, 2.0, SEED);
                MatrixD v = Matrices.randomUniformD(1, s[1], 1.0, 2.0, SEED + 1L);
                MatrixD stretched = Matrices.createD(s[0], s[1]);
                for (int row = 0; row < s[0]; ++row) {
                    stretched.setSubmatrixInplace(row, 0, v, 0, 0, 0, s[1] - 1);
                }
                MatrixD viaVector = applyD(d.copy(), v, op);
                MatrixD viaMatrix = applyD(d.copy(), stretched, op);
                assertBitsD(op + " broadcast vs stretched at " + s[0] + "x" + s[1], viaMatrix.getArrayUnsafe(),
                        viaVector.getArrayUnsafe());
            }
        }
    }

    @Test
    public void testRowBroadcastIsTheTransposeOfColumnBroadcast() {
        for (int[] s : SHAPES) {
            String at = " at " + s[0] + "x" + s[1];
            for (char op : OPS) {
                MatrixD d = Matrices.randomUniformD(s[0], s[1], 1.0, 2.0, SEED);
                MatrixD vd = Matrices.randomUniformD(1, s[1], 1.0, 2.0, SEED + 1L);
                MatrixD viaColumnsD = applyColumnD(d.transpose(), vd.transpose(), op).transpose();
                assertBitsD(op + " MatrixD" + at, viaColumnsD.getArrayUnsafe(),
                        applyD(d, vd, op).getArrayUnsafe());

                MatrixF f = Matrices.randomUniformF(s[0], s[1], 1.0f, 2.0f, SEED);
                MatrixF vf = Matrices.randomUniformF(1, s[1], 1.0f, 2.0f, SEED + 1L);
                MatrixF viaColumnsF = applyColumnF(f.transpose(), vf.transpose(), op).transpose();
                assertBitsF(op + " MatrixF" + at, viaColumnsF.getArrayUnsafe(),
                        applyF(f, vf, op).getArrayUnsafe());
            }
        }
    }

    @Test
    public void testTheCopyVariantsLeaveTheReceiverAlone() {
        for (int[] s : SHAPES) {
            String at = " at " + s[0] + "x" + s[1];
            MatrixD d = Matrices.randomUniformD(s[0], s[1], 1.0, 2.0, SEED);
            MatrixD vd = Matrices.randomUniformD(1, s[1], 1.0, 2.0, SEED + 1L);
            double[] untouchedD = d.getArrayUnsafe().clone();
            MatrixD pd = d.plusBroadcastedRowVector(vd);
            MatrixD md = d.mulBroadcastedRowVector(vd);
            MatrixD qd = d.divBroadcastedRowVector(vd);
            assertBitsD("MatrixD receiver untouched" + at, untouchedD, d.getArrayUnsafe());
            assertNotSame("plusBroadcastedRowVector is a copy" + at, d, pd);
            assertNotSame("mulBroadcastedRowVector is a copy" + at, d, md);
            assertNotSame("divBroadcastedRowVector is a copy" + at, d, qd);
            assertBitsD("plusBroadcastedRowVector" + at,
                    refD(untouchedD, vd.getArrayUnsafe(), s[0], s[1], true, ADD), pd.getArrayUnsafe());
            assertBitsD("mulBroadcastedRowVector" + at,
                    refD(untouchedD, vd.getArrayUnsafe(), s[0], s[1], true, MUL), md.getArrayUnsafe());
            assertBitsD("divBroadcastedRowVector" + at,
                    refD(untouchedD, vd.getArrayUnsafe(), s[0], s[1], true, DIV), qd.getArrayUnsafe());

            MatrixF f = Matrices.randomUniformF(s[0], s[1], 1.0f, 2.0f, SEED);
            MatrixF vf = Matrices.randomUniformF(1, s[1], 1.0f, 2.0f, SEED + 1L);
            float[] untouchedF = f.getArrayUnsafe().clone();
            MatrixF pf = f.plusBroadcastedRowVector(vf);
            MatrixF mf = f.mulBroadcastedRowVector(vf);
            MatrixF qf = f.divBroadcastedRowVector(vf);
            assertBitsF("MatrixF receiver untouched" + at, untouchedF, f.getArrayUnsafe());
            assertNotSame("plusBroadcastedRowVector is a copy" + at, f, pf);
            assertNotSame("mulBroadcastedRowVector is a copy" + at, f, mf);
            assertNotSame("divBroadcastedRowVector is a copy" + at, f, qf);
            assertBitsF("plusBroadcastedRowVector" + at,
                    refF(untouchedF, vf.getArrayUnsafe(), s[0], s[1], true, ADD), pf.getArrayUnsafe());
            assertBitsF("mulBroadcastedRowVector" + at,
                    refF(untouchedF, vf.getArrayUnsafe(), s[0], s[1], true, MUL), mf.getArrayUnsafe());
            assertBitsF("divBroadcastedRowVector" + at,
                    refF(untouchedF, vf.getArrayUnsafe(), s[0], s[1], true, DIV), qf.getArrayUnsafe());
        }
    }

    @Test
    public void testDivisionByZeroFollowsIeee754() {
        double[] numerator = { 1.0, -1.0, 0.0, Double.NaN };
        MatrixD d = Matrices.createD(2, 4);
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 4; ++col) {
                d.set(row, col, numerator[col]);
            }
        }
        d.divBroadcastedRowVectorInplace(Matrices.createD(1, 4));
        // 1/0 = +Inf, -1/0 = -Inf, 0/0 = NaN, NaN/0 = NaN, in both rows
        double[] want = { Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                Double.NEGATIVE_INFINITY, Double.NaN, Double.NaN, Double.NaN, Double.NaN };
        assertBitsD("MatrixD division by zero", want, d.getArrayUnsafe());

        MatrixD negZero = Matrices.createD(1, 4);
        for (int col = 0; col < 4; ++col) {
            negZero.set(0, col, -0.0);
        }
        MatrixD e = Matrices.createD(3, 4);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 4; ++col) {
                e.set(row, col, numerator[col]);
            }
        }
        e.divBroadcastedRowVectorInplace(negZero);
        double[] wantNeg = { Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY,
                Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.NaN,
                Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN };
        assertBitsD("MatrixD division by negative zero", wantNeg, e.getArrayUnsafe());

        MatrixF f = Matrices.createF(2, 2);
        f.set(0, 0, 1.0f);
        f.set(1, 0, 1.0f);
        f.set(0, 1, -1.0f);
        f.set(1, 1, -1.0f);
        f.divBroadcastedRowVectorInplace(Matrices.createF(1, 2));
        float[] wantF = { Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
                Float.NEGATIVE_INFINITY };
        assertBitsF("MatrixF division by zero", wantF, f.getArrayUnsafe());
    }

    @Test
    public void testMultiplicationKeepsTheSignOfZero() {
        MatrixD d = Matrices.createD(2, 2);
        d.set(0, 0, 0.0);
        d.set(1, 0, -0.0);
        d.set(0, 1, 0.0);
        d.set(1, 1, -0.0);
        MatrixD v = Matrices.createD(1, 2);
        v.set(0, 0, -1.0);
        v.set(0, 1, 1.0);
        d.mulBroadcastedRowVectorInplace(v);
        double[] want = { -0.0, 0.0, 0.0, -0.0 };
        assertBitsD("MatrixD signed zero under multiplication", want, d.getArrayUnsafe());
    }

    @Test
    public void testMismatchedDimensionsAreRejected() {
        MatrixD a = Matrices.randomUniformD(4, 3, SEED);
        MatrixD[] badD = { Matrices.randomUniformD(1, 2, SEED), // wrong column count
                Matrices.randomUniformD(2, 3, SEED), // neither equal nor a row vector
                Matrices.randomUniformD(4, 1, SEED) }; // a column vector
        for (MatrixD b : badD) {
            String at = " with " + b.numRows() + "x" + b.numColumns();
            rejects("add" + at, () -> a.addBroadcastedRowVectorInplace(b));
            rejects("mul" + at, () -> a.mulBroadcastedRowVectorInplace(b));
            rejects("div" + at, () -> a.divBroadcastedRowVectorInplace(b));
            rejects("plus copy" + at, () -> a.plusBroadcastedRowVector(b));
            rejects("mul copy" + at, () -> a.mulBroadcastedRowVector(b));
            rejects("div copy" + at, () -> a.divBroadcastedRowVector(b));
        }

        MatrixF c = Matrices.randomUniformF(4, 3, SEED);
        MatrixF[] badF = { Matrices.randomUniformF(1, 2, SEED), Matrices.randomUniformF(2, 3, SEED),
                Matrices.randomUniformF(4, 1, SEED) };
        for (MatrixF b : badF) {
            String at = " MatrixF with " + b.numRows() + "x" + b.numColumns();
            rejects("add" + at, () -> c.addBroadcastedRowVectorInplace(b));
            rejects("mul" + at, () -> c.mulBroadcastedRowVectorInplace(b));
            rejects("div" + at, () -> c.divBroadcastedRowVectorInplace(b));
            rejects("plus copy" + at, () -> c.plusBroadcastedRowVector(b));
            rejects("mul copy" + at, () -> c.mulBroadcastedRowVector(b));
            rejects("div copy" + at, () -> c.divBroadcastedRowVector(b));
        }
    }

    @Test
    public void testTheTwoFamiliesStaySeparate() {
        // a row vector is not silently accepted by the column methods and a
        // column vector is not silently accepted by the row methods
        MatrixD a = Matrices.randomUniformD(4, 3, SEED);
        MatrixD row = Matrices.randomUniformD(1, 3, SEED);
        MatrixD col = Matrices.randomUniformD(4, 1, SEED);
        rejects("column add with a row vector", () -> a.addBroadcastedVectorInplace(row));
        rejects("column mul with a row vector", () -> a.mulBroadcastedVectorInplace(row));
        rejects("column div with a row vector", () -> a.divBroadcastedVectorInplace(row));
        rejects("row add with a column vector", () -> a.addBroadcastedRowVectorInplace(col));
        rejects("row mul with a column vector", () -> a.mulBroadcastedRowVectorInplace(col));
        rejects("row div with a column vector", () -> a.divBroadcastedRowVectorInplace(col));

        MatrixF f = Matrices.randomUniformF(4, 3, SEED);
        MatrixF rowF = Matrices.randomUniformF(1, 3, SEED);
        MatrixF colF = Matrices.randomUniformF(4, 1, SEED);
        rejects("MatrixF column plus with a row vector", () -> f.plusBroadcastedVector(rowF));
        rejects("MatrixF row plus with a column vector", () -> f.plusBroadcastedRowVector(colF));
    }

    @Test
    public void testARejectedCallLeavesTheMatrixUntouched() {
        MatrixD a = Matrices.randomUniformD(4, 3, 1.0, 2.0, SEED);
        double[] untouched = a.getArrayUnsafe().clone();
        MatrixD wrongRows = Matrices.randomUniformD(2, 3, 1.0, 2.0, SEED);
        rejects("add", () -> a.addBroadcastedRowVectorInplace(wrongRows));
        assertBitsD("after a rejected add", untouched, a.getArrayUnsafe());
        rejects("mul", () -> a.mulBroadcastedRowVectorInplace(wrongRows));
        assertBitsD("after a rejected mul", untouched, a.getArrayUnsafe());
        rejects("div", () -> a.divBroadcastedRowVectorInplace(wrongRows));
        assertBitsD("after a rejected div", untouched, a.getArrayUnsafe());
    }

    // ------------------------------------------------------------- helpers

    private static MatrixD applyD(MatrixD A, MatrixD B, char op) {
        switch (op) {
        case ADD:
            return A.addBroadcastedRowVectorInplace(B);
        case MUL:
            return A.mulBroadcastedRowVectorInplace(B);
        default:
            return A.divBroadcastedRowVectorInplace(B);
        }
    }

    private static MatrixF applyF(MatrixF A, MatrixF B, char op) {
        switch (op) {
        case ADD:
            return A.addBroadcastedRowVectorInplace(B);
        case MUL:
            return A.mulBroadcastedRowVectorInplace(B);
        default:
            return A.divBroadcastedRowVectorInplace(B);
        }
    }

    private static MatrixD applyColumnD(MatrixD A, MatrixD B, char op) {
        switch (op) {
        case ADD:
            return A.addBroadcastedVectorInplace(B);
        case MUL:
            return A.mulBroadcastedVectorInplace(B);
        default:
            return A.divBroadcastedVectorInplace(B);
        }
    }

    private static MatrixF applyColumnF(MatrixF A, MatrixF B, char op) {
        switch (op) {
        case ADD:
            return A.addBroadcastedVectorInplace(B);
        case MUL:
            return A.mulBroadcastedVectorInplace(B);
        default:
            return A.divBroadcastedVectorInplace(B);
        }
    }

    /**
     * The expected result, computed the slow and obvious way. {@code broadcast}
     * selects between reading {@code b} as a row vector (one value per column)
     * and reading it as a full matrix.
     */
    private static double[] refD(double[] a, double[] b, int rows, int cols, boolean broadcast, char op) {
        double[] out = a.clone();
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                int i = col * rows + row;
                double y = broadcast ? b[col] : b[i];
                if (op == ADD) {
                    out[i] = out[i] + y;
                } else if (op == MUL) {
                    out[i] = out[i] * y;
                } else {
                    out[i] = out[i] / y;
                }
            }
        }
        return out;
    }

    private static float[] refF(float[] a, float[] b, int rows, int cols, boolean broadcast, char op) {
        float[] out = a.clone();
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                int i = col * rows + row;
                float y = broadcast ? b[col] : b[i];
                if (op == ADD) {
                    out[i] = out[i] + y;
                } else if (op == MUL) {
                    out[i] = out[i] * y;
                } else {
                    out[i] = out[i] / y;
                }
            }
        }
        return out;
    }

    /**
     * Bitwise comparison, but with all NaNs collapsed onto one value: these
     * operations compute, and the bit pattern of an arithmetically produced NaN
     * is not specified. The signed zeros stay distinguishable, which is the
     * point of comparing bits at all here.
     */
    private static void assertBitsD(String what, double[] expected, double[] actual) {
        assertEquals(what + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            long e = Double.doubleToLongBits(expected[i]);
            long a = Double.doubleToLongBits(actual[i]);
            if (e != a) {
                fail(what + " differs at " + i + " : expected " + expected[i] + " but was " + actual[i]);
            }
        }
    }

    private static void assertBitsF(String what, float[] expected, float[] actual) {
        assertEquals(what + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            int e = Float.floatToIntBits(expected[i]);
            int a = Float.floatToIntBits(actual[i]);
            if (e != a) {
                fail(what + " differs at " + i + " : expected " + expected[i] + " but was " + actual[i]);
            }
        }
    }

    private interface Body {
        void run();
    }

    private static void rejects(String what, Body body) {
        try {
            body.run();
            fail(what + " : expected an IndexOutOfBoundsException but none was thrown");
        } catch (IndexOutOfBoundsException expected) {
            // the guard did its job
        }
    }
}
