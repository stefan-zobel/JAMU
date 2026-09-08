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
 * No decomposition may overwrite the matrix it was given. The complex SVD did,
 * in every entry of it, which is what this test was written for; the real side
 * is here too, so that a regression in either direction shows up.
 */
public final class DecompositionInputTest {

    private static final long SEED = 42L;

    /** wide, tall and quadratic, because the routines branch on the shape */
    private static final int[][] SHAPES = { { 40, 30 }, { 30, 40 }, { 25, 25 } };

    /** everything a caller can reach that runs a decomposition underneath */
    private static String[] entryPoints(int m, int n) {
        if (m == n) {
            return new String[] { "svd(true)", "svd(false)", "svdEcon()", "norm2()",
                    "singularValues()", "pseudoInv()", "numericalRank()", "evd(true)",
                    "evd(false)", "lud()", "qrd()", "solve()", "inv()" };
        }
        if (m > n) {
            return new String[] { "svd(true)", "svd(false)", "svdEcon()", "norm2()",
                    "singularValues()", "pseudoInv()", "numericalRank()", "lud()", "qrd()" };
        }
        return new String[] { "svd(true)", "svd(false)", "svdEcon()", "norm2()",
                "singularValues()", "pseudoInv()", "numericalRank()", "lud()" };
    }

    private static void applyComplexD(String op, ComplexMatrixD a) {
        int m = a.numRows();
        int n = a.numColumns();
        if (op.equals("svd(true)")) {
            a.svd(true);
        } else if (op.equals("svd(false)")) {
            a.svd(false);
        } else if (op.equals("svdEcon()")) {
            a.svdEcon();
        } else if (op.equals("norm2()")) {
            a.norm2();
        } else if (op.equals("singularValues()")) {
            a.singularValues();
        } else if (op.equals("pseudoInv()")) {
            a.pseudoInv();
        } else if (op.equals("numericalRank()")) {
            Matrices.numericalRank(a);
        } else if (op.equals("evd(true)")) {
            a.evd(true);
        } else if (op.equals("evd(false)")) {
            a.evd(false);
        } else if (op.equals("lud()")) {
            a.lud();
        } else if (op.equals("qrd()")) {
            a.qrd();
        } else if (op.equals("solve()")) {
            a.solve(Matrices.identityComplexD(n), Matrices.createComplexD(n, n));
        } else {
            a.inv(Matrices.createComplexD(m, n));
        }
    }

    private static void applyComplexF(String op, ComplexMatrixF a) {
        int m = a.numRows();
        int n = a.numColumns();
        if (op.equals("svd(true)")) {
            a.svd(true);
        } else if (op.equals("svd(false)")) {
            a.svd(false);
        } else if (op.equals("svdEcon()")) {
            a.svdEcon();
        } else if (op.equals("norm2()")) {
            a.norm2();
        } else if (op.equals("singularValues()")) {
            a.singularValues();
        } else if (op.equals("pseudoInv()")) {
            a.pseudoInv();
        } else if (op.equals("numericalRank()")) {
            Matrices.numericalRank(a);
        } else if (op.equals("evd(true)")) {
            a.evd(true);
        } else if (op.equals("evd(false)")) {
            a.evd(false);
        } else if (op.equals("lud()")) {
            a.lud();
        } else if (op.equals("qrd()")) {
            a.qrd();
        } else if (op.equals("solve()")) {
            a.solve(Matrices.identityComplexF(n), Matrices.createComplexF(n, n));
        } else {
            a.inv(Matrices.createComplexF(m, n));
        }
    }

    private static void applyD(String op, MatrixD a) {
        int m = a.numRows();
        int n = a.numColumns();
        if (op.equals("svd(true)")) {
            a.svd(true);
        } else if (op.equals("svd(false)")) {
            a.svd(false);
        } else if (op.equals("svdEcon()")) {
            a.svdEcon();
        } else if (op.equals("norm2()")) {
            a.norm2();
        } else if (op.equals("singularValues()")) {
            a.singularValues();
        } else if (op.equals("pseudoInv()")) {
            a.pseudoInv();
        } else if (op.equals("numericalRank()")) {
            Matrices.numericalRank(a);
        } else if (op.equals("evd(true)")) {
            a.evd(true);
        } else if (op.equals("evd(false)")) {
            a.evd(false);
        } else if (op.equals("lud()")) {
            a.lud();
        } else if (op.equals("qrd()")) {
            a.qrd();
        } else if (op.equals("solve()")) {
            a.solve(Matrices.identityD(n), Matrices.createD(n, n));
        } else {
            a.inv(Matrices.createD(m, n));
        }
    }

    private static void applyF(String op, MatrixF a) {
        int m = a.numRows();
        int n = a.numColumns();
        if (op.equals("svd(true)")) {
            a.svd(true);
        } else if (op.equals("svd(false)")) {
            a.svd(false);
        } else if (op.equals("svdEcon()")) {
            a.svdEcon();
        } else if (op.equals("norm2()")) {
            a.norm2();
        } else if (op.equals("singularValues()")) {
            a.singularValues();
        } else if (op.equals("pseudoInv()")) {
            a.pseudoInv();
        } else if (op.equals("numericalRank()")) {
            Matrices.numericalRank(a);
        } else if (op.equals("evd(true)")) {
            a.evd(true);
        } else if (op.equals("evd(false)")) {
            a.evd(false);
        } else if (op.equals("lud()")) {
            a.lud();
        } else if (op.equals("qrd()")) {
            a.qrd();
        } else if (op.equals("solve()")) {
            a.solve(Matrices.identityF(n), Matrices.createF(n, n));
        } else {
            a.inv(Matrices.createF(m, n));
        }
    }

    /** bit for bit, because a decomposition has no business rounding the input */
    private static void untouched(String what, double[] before, double[] after) {
        assertEquals(what + ": the array changed length", before.length, after.length);
        for (int i = 0; i < before.length; ++i) {
            if (Double.doubleToLongBits(before[i]) != Double.doubleToLongBits(after[i])) {
                int changed = 0;
                for (int j = 0; j < before.length; ++j) {
                    if (Double.doubleToLongBits(before[j]) != Double.doubleToLongBits(after[j])) {
                        ++changed;
                    }
                }
                assertEquals(what + " overwrote the caller's matrix, entry " + i + " and "
                        + (changed - 1) + " more of " + before.length, before[i], after[i], 0.0);
            }
        }
    }

    private static void untouched(String what, float[] before, float[] after) {
        assertEquals(what + ": the array changed length", before.length, after.length);
        for (int i = 0; i < before.length; ++i) {
            if (Float.floatToIntBits(before[i]) != Float.floatToIntBits(after[i])) {
                int changed = 0;
                for (int j = 0; j < before.length; ++j) {
                    if (Float.floatToIntBits(before[j]) != Float.floatToIntBits(after[j])) {
                        ++changed;
                    }
                }
                assertEquals(what + " overwrote the caller's matrix, entry " + i + " and "
                        + (changed - 1) + " more of " + before.length, before[i], after[i], 0.0f);
            }
        }
    }

    private static String at(String op, int m, int n) {
        return op + " at " + m + "x" + n;
    }

    @Test
    public void testTheComplexDoubleDecompositionsLeaveTheirInputAlone() {
        int seen = 0;
        for (int[] s : SHAPES) {
            for (String op : entryPoints(s[0], s[1])) {
                ComplexMatrixD a = Matrices.randomUniformComplexD(s[0], s[1], SEED);
                double[] before = a.getArrayUnsafe().clone();
                applyComplexD(op, a);
                untouched("ComplexMatrixD " + at(op, s[0], s[1]), before, a.getArrayUnsafe());
                ++seen;
            }
        }
        assertEquals("entry points covered", 30, seen);
    }

    @Test
    public void testTheComplexFloatDecompositionsLeaveTheirInputAlone() {
        for (int[] s : SHAPES) {
            for (String op : entryPoints(s[0], s[1])) {
                ComplexMatrixF a = Matrices.randomUniformComplexF(s[0], s[1], SEED);
                float[] before = a.getArrayUnsafe().clone();
                applyComplexF(op, a);
                untouched("ComplexMatrixF " + at(op, s[0], s[1]), before, a.getArrayUnsafe());
            }
        }
    }

    @Test
    public void testTheRealDecompositionsLeaveTheirInputAlone() {
        for (int[] s : SHAPES) {
            for (String op : entryPoints(s[0], s[1])) {
                MatrixD a = Matrices.randomUniformD(s[0], s[1], SEED);
                double[] before = a.getArrayUnsafe().clone();
                applyD(op, a);
                untouched("MatrixD " + at(op, s[0], s[1]), before, a.getArrayUnsafe());
                MatrixF b = Matrices.randomUniformF(s[0], s[1], SEED);
                float[] fb = b.getArrayUnsafe().clone();
                applyF(op, b);
                untouched("MatrixF " + at(op, s[0], s[1]), fb, b.getArrayUnsafe());
            }
        }
    }

    @Test
    public void testTheSvdStillFactorsTheMatrixItWasGiven() {
        // the copy has to carry the right content, not an empty matrix, and
        // this is what says so
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            ComplexMatrixD a = Matrices.randomUniformComplexD(m, n, SEED);
            SvdComplexD svd = a.svd(true);
            double[] sv = svd.getS();
            ComplexMatrixD sigma = Matrices.createComplexD(m, n);
            for (int i = 0; i < sv.length; ++i) {
                sigma.set(i, i, sv[i], 0.0);
            }
            ComplexMatrixD us = svd.getU().mult(sigma, Matrices.createComplexD(m, n));
            ComplexMatrixD back = us.mult(svd.getVh(), Matrices.createComplexD(m, n));
            double err = back.addInplace(-1.0, 0.0, a).normF() / a.normF();
            assertTrue("U S Vh is off by " + err + " at " + m + "x" + n, err <= 1.0e-13);
            // and the matrix is still there afterwards, which is the point
            ComplexMatrixD again = Matrices.randomUniformComplexD(m, n, SEED);
            untouched("the factored matrix at " + m + "x" + n, again.getArrayUnsafe(),
                    a.getArrayUnsafe());
        }
    }

    @Test
    public void testTheSingularValuesAreWhatTheyWere() {
        // routing through a copy hands zgesdd the same bytes, so the singular
        // values must not move at all
        for (int[] s : SHAPES) {
            ComplexMatrixD a = Matrices.randomUniformComplexD(s[0], s[1], SEED);
            double[] one = a.svd(false).getS();
            double[] two = a.copy().svd(false).getS();
            for (int i = 0; i < one.length; ++i) {
                assertEquals("singular value " + i + " at " + s[0] + "x" + s[1],
                        Double.doubleToLongBits(one[i]), Double.doubleToLongBits(two[i]));
            }
        }
    }

    @Test
    public void testTheRightSingularVectorsAreNotBitReproducible() {
        // MKL's ?gesdd does not return the same V twice for the same input, so
        // no test may compare that factor bit for bit. The values agree to the
        // last bit or two; it is worth knowing before someone writes a test
        // that fails once a week
        ComplexMatrixD a = Matrices.randomUniformComplexD(40, 30, SEED);
        double[] first = a.copy().svd(true).getVh().getArrayUnsafe().clone();
        double worst = 0.0;
        for (int i = 0; i < 8; ++i) {
            double[] again = a.copy().svd(true).getVh().getArrayUnsafe();
            for (int j = 0; j < first.length; ++j) {
                worst = Math.max(worst, Math.abs(first[j] - again[j]));
            }
        }
        assertTrue("the right singular vectors moved by " + worst, worst <= 1.0e-14);
        // the singular values, on the other hand, are stable
        double[] s1 = a.copy().svd(true).getS();
        double[] s2 = a.copy().svd(true).getS();
        for (int i = 0; i < s1.length; ++i) {
            assertEquals("singular value " + i, Double.doubleToLongBits(s1[i]),
                    Double.doubleToLongBits(s2[i]));
        }
    }
}
