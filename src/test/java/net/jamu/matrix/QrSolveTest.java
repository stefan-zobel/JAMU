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
import static org.junit.Assert.fail;

import java.util.Random;

import org.junit.Test;

import net.frobenius.TTrans;
import net.frobenius.lapack.PlainLapack;

/**
 * Pins {@code solve} on non-square matrices against a direct {@code *gels} call
 * in all four matrix hierarchies.
 */
public final class QrSolveTest {

    private static final long SEED = 20260918L;

    /** (rows, cols, rhs): small, one copy at least 40 rows high, both copies */
    private static final int[][] SHAPES = { { 7, 4, 3 }, { 4, 7, 3 }, { 60, 5, 2 }, { 5, 60, 2 }, { 50, 45, 2 },
            { 45, 50, 2 } };

    /** (rows, cols, rhs) for an output view */
    private static final int[][] VIEW_SHAPES = { { 7, 4, 3 }, { 45, 50, 2 } };

    @Test
    public void testMatrixDSolveMatchesDgels() {
        Random r = new Random(SEED);
        for (int[] s : SHAPES) {
            MatrixD A = randomD(r, s[0], s[1]);
            MatrixD B = randomD(r, s[0], s[2]);
            MatrixD a0 = A.copy();
            double[] expected = referenceD(A, B);
            assertClose(label("B", s), expected, A.solve(B, Matrices.createD(s[1], s[2])).getArrayUnsafe());
            MatrixD parent = randomD(r, s[0] + 3, s[2] + 2);
            parent.setSubmatrixInplace(2, 1, B, 0, 0, s[0] - 1, s[2] - 1);
            MatrixD view = Matrices.view(parent, 2, 1, s[0] + 1, s[2]);
            assertClose(label("view B", s), expected, A.solve(view, Matrices.createD(s[1], s[2])).getArrayUnsafe());
            assertBits(label("A", s), a0.getArrayUnsafe(), A.getArrayUnsafe());
        }
    }

    @Test
    public void testMatrixFSolveMatchesSgels() {
        Random r = new Random(SEED);
        for (int[] s : SHAPES) {
            MatrixF A = randomF(r, s[0], s[1]);
            MatrixF B = randomF(r, s[0], s[2]);
            MatrixF a0 = A.copy();
            float[] expected = referenceF(A, B);
            assertClose(label("B", s), expected, A.solve(B, Matrices.createF(s[1], s[2])).getArrayUnsafe());
            MatrixF parent = randomF(r, s[0] + 3, s[2] + 2);
            parent.setSubmatrixInplace(2, 1, B, 0, 0, s[0] - 1, s[2] - 1);
            MatrixF view = Matrices.view(parent, 2, 1, s[0] + 1, s[2]);
            assertClose(label("view B", s), expected, A.solve(view, Matrices.createF(s[1], s[2])).getArrayUnsafe());
            assertBits(label("A", s), a0.getArrayUnsafe(), A.getArrayUnsafe());
        }
    }

    @Test
    public void testComplexMatrixDSolveMatchesZgels() {
        Random r = new Random(SEED);
        for (int[] s : SHAPES) {
            ComplexMatrixD A = randomComplexD(r, s[0], s[1]);
            ComplexMatrixD B = randomComplexD(r, s[0], s[2]);
            ComplexMatrixD a0 = A.copy();
            double[] expected = referenceComplexD(A, B);
            assertClose(label("B", s), expected, A.solve(B, Matrices.createComplexD(s[1], s[2])).getArrayUnsafe());
            ComplexMatrixD parent = randomComplexD(r, s[0] + 3, s[2] + 2);
            parent.setSubmatrixInplace(2, 1, B, 0, 0, s[0] - 1, s[2] - 1);
            ComplexMatrixD view = Matrices.view(parent, 2, 1, s[0] + 1, s[2]);
            assertClose(label("view B", s), expected,
                    A.solve(view, Matrices.createComplexD(s[1], s[2])).getArrayUnsafe());
            assertBits(label("A", s), a0.getArrayUnsafe(), A.getArrayUnsafe());
        }
    }

    @Test
    public void testComplexMatrixFSolveMatchesCgels() {
        Random r = new Random(SEED);
        for (int[] s : SHAPES) {
            ComplexMatrixF A = randomComplexF(r, s[0], s[1]);
            ComplexMatrixF B = randomComplexF(r, s[0], s[2]);
            ComplexMatrixF a0 = A.copy();
            float[] expected = referenceComplexF(A, B);
            assertClose(label("B", s), expected, A.solve(B, Matrices.createComplexF(s[1], s[2])).getArrayUnsafe());
            ComplexMatrixF parent = randomComplexF(r, s[0] + 3, s[2] + 2);
            parent.setSubmatrixInplace(2, 1, B, 0, 0, s[0] - 1, s[2] - 1);
            ComplexMatrixF view = Matrices.view(parent, 2, 1, s[0] + 1, s[2]);
            assertClose(label("view B", s), expected,
                    A.solve(view, Matrices.createComplexF(s[1], s[2])).getArrayUnsafe());
            assertBits(label("A", s), a0.getArrayUnsafe(), A.getArrayUnsafe());
        }
    }

    @Test
    public void testSolveIntoAViewThrowsAndLeavesTheParentUnchanged() {
        Random r = new Random(SEED);
        for (int[] s : VIEW_SHAPES) {
            MatrixD pd = randomD(r, s[1] + 2, s[2] + 1);
            double[] pd0 = pd.getArrayUnsafe().clone();
            assertRefused(label("MatrixD", s), () -> randomD(r, s[0], s[1]).solve(randomD(r, s[0], s[2]),
                    Matrices.view(pd, 1, 1, s[1], s[2])));
            assertBits(label("MatrixD parent", s), pd0, pd.getArrayUnsafe());

            MatrixF pf = randomF(r, s[1] + 2, s[2] + 1);
            float[] pf0 = pf.getArrayUnsafe().clone();
            assertRefused(label("MatrixF", s), () -> randomF(r, s[0], s[1]).solve(randomF(r, s[0], s[2]),
                    Matrices.view(pf, 1, 1, s[1], s[2])));
            assertBits(label("MatrixF parent", s), pf0, pf.getArrayUnsafe());

            ComplexMatrixD pzd = randomComplexD(r, s[1] + 2, s[2] + 1);
            double[] pzd0 = pzd.getArrayUnsafe().clone();
            assertRefused(label("ComplexMatrixD", s), () -> randomComplexD(r, s[0], s[1])
                    .solve(randomComplexD(r, s[0], s[2]), Matrices.view(pzd, 1, 1, s[1], s[2])));
            assertBits(label("ComplexMatrixD parent", s), pzd0, pzd.getArrayUnsafe());

            ComplexMatrixF pzf = randomComplexF(r, s[1] + 2, s[2] + 1);
            float[] pzf0 = pzf.getArrayUnsafe().clone();
            assertRefused(label("ComplexMatrixF", s), () -> randomComplexF(r, s[0], s[1])
                    .solve(randomComplexF(r, s[0], s[2]), Matrices.view(pzf, 1, 1, s[1], s[2])));
            assertBits(label("ComplexMatrixF parent", s), pzf0, pzf.getArrayUnsafe());
        }
    }

    // ---------------------------------------------------------------- references

    static double[] referenceD(MatrixD A, MatrixD B) {
        int mm = A.numRows();
        int nn = A.numColumns();
        int rhs = B.numColumns();
        int ld = Math.max(mm, nn);
        double[] b = B.getArrayUnsafe();
        double[] tmp = new double[ld * rhs];
        for (int j = 0; j < rhs; ++j) {
            for (int i = 0; i < mm; ++i) {
                tmp[j * ld + i] = b[j * mm + i];
            }
        }
        PlainLapack.dgels(Matrices.getLapack(), TTrans.NO_TRANS, mm, nn, rhs, A.getArrayUnsafe().clone(), mm, tmp,
                ld);
        double[] x = new double[nn * rhs];
        for (int j = 0; j < rhs; ++j) {
            for (int i = 0; i < nn; ++i) {
                x[j * nn + i] = tmp[j * ld + i];
            }
        }
        return x;
    }

    static float[] referenceF(MatrixF A, MatrixF B) {
        int mm = A.numRows();
        int nn = A.numColumns();
        int rhs = B.numColumns();
        int ld = Math.max(mm, nn);
        float[] b = B.getArrayUnsafe();
        float[] tmp = new float[ld * rhs];
        for (int j = 0; j < rhs; ++j) {
            for (int i = 0; i < mm; ++i) {
                tmp[j * ld + i] = b[j * mm + i];
            }
        }
        PlainLapack.sgels(Matrices.getLapack(), TTrans.NO_TRANS, mm, nn, rhs, A.getArrayUnsafe().clone(), mm, tmp,
                ld);
        float[] x = new float[nn * rhs];
        for (int j = 0; j < rhs; ++j) {
            for (int i = 0; i < nn; ++i) {
                x[j * nn + i] = tmp[j * ld + i];
            }
        }
        return x;
    }

    static double[] referenceComplexD(ComplexMatrixD A, ComplexMatrixD B) {
        int mm = A.numRows();
        int nn = A.numColumns();
        int rhs = B.numColumns();
        int ld = Math.max(mm, nn);
        double[] b = B.getArrayUnsafe();
        double[] tmp = new double[2 * ld * rhs];
        for (int j = 0; j < rhs; ++j) {
            for (int i = 0; i < mm; ++i) {
                tmp[2 * (j * ld + i)] = b[2 * (j * mm + i)];
                tmp[2 * (j * ld + i) + 1] = b[2 * (j * mm + i) + 1];
            }
        }
        PlainLapack.zgels(Matrices.getLapack(), TTrans.NO_TRANS, mm, nn, rhs, A.getArrayUnsafe().clone(), mm, tmp,
                ld);
        double[] x = new double[2 * nn * rhs];
        for (int j = 0; j < rhs; ++j) {
            for (int i = 0; i < nn; ++i) {
                x[2 * (j * nn + i)] = tmp[2 * (j * ld + i)];
                x[2 * (j * nn + i) + 1] = tmp[2 * (j * ld + i) + 1];
            }
        }
        return x;
    }

    static float[] referenceComplexF(ComplexMatrixF A, ComplexMatrixF B) {
        int mm = A.numRows();
        int nn = A.numColumns();
        int rhs = B.numColumns();
        int ld = Math.max(mm, nn);
        float[] b = B.getArrayUnsafe();
        float[] tmp = new float[2 * ld * rhs];
        for (int j = 0; j < rhs; ++j) {
            for (int i = 0; i < mm; ++i) {
                tmp[2 * (j * ld + i)] = b[2 * (j * mm + i)];
                tmp[2 * (j * ld + i) + 1] = b[2 * (j * mm + i) + 1];
            }
        }
        PlainLapack.cgels(Matrices.getLapack(), TTrans.NO_TRANS, mm, nn, rhs, A.getArrayUnsafe().clone(), mm, tmp,
                ld);
        float[] x = new float[2 * nn * rhs];
        for (int j = 0; j < rhs; ++j) {
            for (int i = 0; i < nn; ++i) {
                x[2 * (j * nn + i)] = tmp[2 * (j * ld + i)];
                x[2 * (j * nn + i) + 1] = tmp[2 * (j * ld + i) + 1];
            }
        }
        return x;
    }

    // ---------------------------------------------------------------- helpers

    static MatrixD randomD(Random r, int rows, int cols) {
        MatrixD m = Matrices.createD(rows, cols);
        double[] a = m.getArrayUnsafe();
        for (int i = 0; i < a.length; ++i) {
            a[i] = r.nextDouble() - 0.5;
        }
        return m;
    }

    static MatrixF randomF(Random r, int rows, int cols) {
        MatrixF m = Matrices.createF(rows, cols);
        float[] a = m.getArrayUnsafe();
        for (int i = 0; i < a.length; ++i) {
            a[i] = r.nextFloat() - 0.5f;
        }
        return m;
    }

    static ComplexMatrixD randomComplexD(Random r, int rows, int cols) {
        ComplexMatrixD m = Matrices.createComplexD(rows, cols);
        double[] a = m.getArrayUnsafe();
        for (int i = 0; i < a.length; ++i) {
            a[i] = r.nextDouble() - 0.5;
        }
        return m;
    }

    static ComplexMatrixF randomComplexF(Random r, int rows, int cols) {
        ComplexMatrixF m = Matrices.createComplexF(rows, cols);
        float[] a = m.getArrayUnsafe();
        for (int i = 0; i < a.length; ++i) {
            a[i] = r.nextFloat() - 0.5f;
        }
        return m;
    }

    static String label(String what, int[] s) {
        return what + " (" + s[0] + " x " + s[1] + ", rhs " + s[2] + ")";
    }

    static void assertRefused(String label, Runnable solve) {
        try {
            solve.run();
            fail(label + ": expected UnsupportedOperationException");
        } catch (UnsupportedOperationException expected) {
            // refused
        }
    }

    // BLAS and LAPACK may round differently for a different memory alignment
    static void assertClose(String label, double[] expected, double[] actual) {
        assertEquals(label + ": length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            double scale = Math.max(1.0, Math.max(Math.abs(expected[i]), Math.abs(actual[i])));
            if (!(Math.abs(expected[i] - actual[i]) <= 1e-10 * scale)) {
                fail(label + ": index " + i + " expected " + expected[i] + " but was " + actual[i]);
            }
        }
    }

    static void assertClose(String label, float[] expected, float[] actual) {
        assertEquals(label + ": length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            float scale = Math.max(1.0f, Math.max(Math.abs(expected[i]), Math.abs(actual[i])));
            if (!(Math.abs(expected[i] - actual[i]) <= 1e-4f * scale)) {
                fail(label + ": index " + i + " expected " + expected[i] + " but was " + actual[i]);
            }
        }
    }

    static void assertBits(String label, double[] expected, double[] actual) {
        assertEquals(label + ": length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            if (Double.doubleToRawLongBits(expected[i]) != Double.doubleToRawLongBits(actual[i])) {
                fail(label + ": index " + i + " expected " + expected[i] + " but was " + actual[i]);
            }
        }
    }

    static void assertBits(String label, float[] expected, float[] actual) {
        assertEquals(label + ": length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            if (Float.floatToRawIntBits(expected[i]) != Float.floatToRawIntBits(actual[i])) {
                fail(label + ": index " + i + " expected " + expected[i] + " but was " + actual[i]);
            }
        }
    }
}
