/*
 * Copyright 2020, 2025 Stefan Zobel
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

import static org.junit.Assert.assertTrue;
import org.junit.Test;

/**
 * Tests for Expm
 */
public final class ExpmTest {

    @Test
    public void testComplex() {
        ComplexMatrixD C = Matrices.createComplexD(2, 2);
        C.set(0, 0, 1.0, 0.0);
        C.set(1, 0, 2.0, 0.0);
        C.set(0, 1, -1.0, 0.0);
        C.set(1, 1, 3.0, 0.0);
        // C = i * C
        C = C.scaleInplace(0.0, 1.0);
        // C = e^C
        C = C.expm();

        // expected
        // reference: scipy.linalg.expm()
        ComplexMatrixD expm = Matrices.createComplexD(2, 2);
        expm.set(0, 0, 4.2645929667E-01, 1.8921755097E+00);
        expm.set(1, 0, -2.1372148428E+00, -9.7811251808E-01);
        expm.set(0, 1, 1.0686074214E+00, 4.8905625904E-01);
        expm.set(1, 1, -1.7107555461E+00, 9.1406299158E-01);
        assertTrue(Matrices.approxEqual(expm, C));
    }

    @Test
    public void testComplexZero() {
        ComplexMatrixD zeros = Matrices.createComplexD(3, 3);
        ComplexMatrixD identity = Matrices.identityComplexD(3);
        assertTrue(Matrices.approxEqual(identity, zeros.expm()));
    }

    @Test
    public void testRealZero() {
        MatrixD zeros = Matrices.createD(3, 3);
        MatrixD identity = Matrices.identityD(3);
        assertTrue(Matrices.approxEqual(identity, zeros.expm()));
    }

    @Test
    public void testRealMatlab() {
        // https://www.mathworks.com/help/matlab/ref/expm.html
        MatrixD A = Matrices.createD(3, 3);
        A.set(0, 0, 1);
        A.set(0, 1, 1);
        A.set(0, 2, 0);
        A.set(1, 0, 0);
        A.set(1, 1, 0);
        A.set(1, 2, 2);
        A.set(2, 0, 0);
        A.set(2, 1, 0);
        A.set(2, 2, -1);
        MatrixD eA = A.expm();

        MatrixD expm = Matrices.createD(3, 3);
        expm.set(0, 0, 2.7183);
        expm.set(0, 1, 1.7183);
        expm.set(0, 2, 1.0862);
        expm.set(1, 0, 0.0);
        expm.set(1, 1, 1.0);
        expm.set(1, 2, 1.2642);
        expm.set(2, 0, 0.0);
        expm.set(2, 1, 0.0);
        expm.set(2, 2, 0.3679);
        assertTrue(Matrices.approxEqual(expm, eA, 1.0e-4));
    }

    @Test
    public void testRealJuliaA1() {
        // A1: https://gist.github.com/sdewaele/2c176cb634280cf8a23c5970739cea0e
        MatrixD A1 = Matrices.createD(3, 3);
        A1.set(0, 0, 4);
        A1.set(0, 1, 2);
        A1.set(0, 2, 0);
        A1.set(1, 0, 1);
        A1.set(1, 1, 4);
        A1.set(1, 2, 1);
        A1.set(2, 0, 1);
        A1.set(2, 1, 1);
        A1.set(2, 2, 4);
        A1 = A1.transpose();
        MatrixD eA1 = A1.expm();

        MatrixD expm = Matrices.createD(3, 3);
        expm.set(0, 0, 147.866622446369);
        expm.set(0, 1, 127.781085523181);
        expm.set(0, 2, 127.781085523182);
        expm.set(1, 0, 183.765138646367);
        expm.set(1, 1, 183.765138646366);
        expm.set(1, 2, 163.679601723179);
        expm.set(2, 0, 71.797032399996);
        expm.set(2, 1, 91.8825693231832);
        expm.set(2, 2, 111.968106246371);
        assertTrue(Matrices.approxEqual(expm, eA1));
    }

    @Test
    public void testRealJuliaA3() {
        // A3: https://gist.github.com/sdewaele/2c176cb634280cf8a23c5970739cea0e
        MatrixD A3 = Matrices.createD(3, 3);
        A3.set(0, 0, -131);
        A3.set(0, 1, 19);
        A3.set(0, 2, 18);
        A3.set(1, 0, -390);
        A3.set(1, 1, 56);
        A3.set(1, 2, 54);
        A3.set(2, 0, -387);
        A3.set(2, 1, 57);
        A3.set(2, 2, 52);
        A3 = A3.transpose();
        MatrixD eA3 = A3.expm();
        
        MatrixD expm = Matrices.createD(3, 3);
        expm.set(0, 0, -1.50964415879218);
        expm.set(0, 1, -5.6325707998812);
        expm.set(0, 2, -4.934938326092);
        expm.set(1, 0, 0.367879439109187);
        expm.set(1, 1, 1.47151775849686);
        expm.set(1, 2, 1.10363831732856);
        expm.set(2, 0, 0.135335281175235);
        expm.set(2, 1, 0.406005843524598);
        expm.set(2, 2, 0.541341126763207);
        assertTrue(Matrices.approxEqual(expm, eA3));
    }

    @Test
    public void testRealJuliaA4() {
        // A4: https://gist.github.com/sdewaele/2c176cb634280cf8a23c5970739cea0e
        MatrixD A4 = Matrices.createD(2, 2);
        A4.set(0, 0, 0.25);
        A4.set(0, 1, 0.25);
        A4.set(1, 0, 0);
        A4.set(1, 1, 0);
        MatrixD eA4 = A4.expm();

        MatrixD expm = Matrices.createD(2, 2);
        expm.set(0, 0, 1.2840254166877416);
        expm.set(0, 1, 0.2840254166877415);
        expm.set(1, 0, 0.0);
        expm.set(1, 1, 1.0);
        assertTrue(Matrices.approxEqual(expm, eA4));
    }

    @Test
    public void testRealJuliaA5() {
        // A5: https://gist.github.com/sdewaele/2c176cb634280cf8a23c5970739cea0e
        MatrixD A5 = Matrices.createD(2, 2);
        A5.set(0, 0, 0);
        A5.set(0, 1, 0.02);
        A5.set(1, 0, 0);
        A5.set(1, 1, 0);
        MatrixD eA5 = A5.expm();

        MatrixD expm = Matrices.createD(2, 2);
        expm.set(0, 0, 1.0);
        expm.set(0, 1, 0.02);
        expm.set(1, 0, 0.0);
        expm.set(1, 1, 1.0);
        assertTrue(Matrices.approxEqual(expm, eA5));
    }

    // exp(diag(x, x)) is exactly diag(exp(x), exp(x)); measured worst is
    // 6.8e-15 in double and 4.3e-6 in single, see TODO.md
    private static final double[] NEGATIVE_D = { -0.1, -1.0, -5.0, -20.0, -50.0 };
    private static final float[] NEGATIVE_F = { -0.1f, -1.0f, -5.0f, -50.0f };

    private static final int[] ONES_DIMS = { 2, 3, 5, 10, 20, 40 };

    private static final double TOL_D = 1.0e-12;
    private static final float TOL_F = 1.0e-4f;

    @Test
    public void testNegativeDiagonalReal() {
        for (double x : NEGATIVE_D) {
            MatrixD A = Matrices.createD(2, 2);
            A.set(0, 0, x);
            A.set(1, 1, x);
            assertRelD("MatrixD at " + x, Math.exp(x), A.expm().get(0, 0), TOL_D);
        }
        for (float x : NEGATIVE_F) {
            MatrixF A = Matrices.createF(2, 2);
            A.set(0, 0, x);
            A.set(1, 1, x);
            assertRelF("MatrixF at " + x, (float) Math.exp(x), A.expm().get(0, 0), TOL_F);
        }
    }

    @Test
    public void testNegativeDiagonalComplex() {
        for (double x : NEGATIVE_D) {
            ComplexMatrixD A = Matrices.createComplexD(2, 2);
            A.set(0, 0, x, 0.0);
            A.set(1, 1, x, 0.0);
            assertRelD("ComplexMatrixD at " + x, Math.exp(x), A.expm().get(0, 0).re(), TOL_D);
        }
        for (float x : NEGATIVE_F) {
            ComplexMatrixF A = Matrices.createComplexF(2, 2);
            A.set(0, 0, x, 0.0f);
            A.set(1, 1, x, 0.0f);
            assertRelF("ComplexMatrixF at " + x, (float) Math.exp(x), A.expm().get(0, 0).re(), TOL_F);
        }
    }

    // exp(ones(n, n)) is I + ((e^n - 1) / n) * ones. The max norm of ones(n, n)
    // is 1 whatever n is, so this is the case a submultiplicative norm is
    // needed for
    @Test
    public void testOnesMatrixReal() {
        for (int n : ONES_DIMS) {
            double off = Math.expm1(n) / n;
            MatrixD eA = Matrices.onesD(n, n).expm();
            assertRelD("MatrixD diagonal at n = " + n, 1.0 + off, eA.get(0, 0), TOL_D);
            assertRelD("MatrixD off-diagonal at n = " + n, off, eA.get(0, 1), TOL_D);
            MatrixF eB = Matrices.onesF(n, n).expm();
            assertRelF("MatrixF diagonal at n = " + n, (float) (1.0 + off), eB.get(0, 0), TOL_F);
            assertRelF("MatrixF off-diagonal at n = " + n, (float) off, eB.get(0, 1), TOL_F);
        }
    }

    @Test
    public void testOnesMatrixComplex() {
        for (int n : ONES_DIMS) {
            double off = Math.expm1(n) / n;
            ComplexMatrixD eA = onesComplexD(n).expm();
            assertRelD("ComplexMatrixD diagonal at n = " + n, 1.0 + off, eA.get(0, 0).re(), TOL_D);
            assertRelD("ComplexMatrixD off-diagonal at n = " + n, off, eA.get(0, 1).re(), TOL_D);
            ComplexMatrixF eB = onesComplexF(n).expm();
            assertRelF("ComplexMatrixF diagonal at n = " + n, (float) (1.0 + off), eB.get(0, 0).re(), TOL_F);
            assertRelF("ComplexMatrixF off-diagonal at n = " + n, (float) off, eB.get(0, 1).re(), TOL_F);
        }
    }

    private static ComplexMatrixD onesComplexD(int n) {
        ComplexMatrixD m = Matrices.createComplexD(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                m.set(i, j, 1.0, 0.0);
            }
        }
        return m;
    }

    private static ComplexMatrixF onesComplexF(int n) {
        ComplexMatrixF m = Matrices.createComplexF(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                m.set(i, j, 1.0f, 0.0f);
            }
        }
        return m;
    }

    private static void assertRelD(String what, double expected, double actual, double tol) {
        double err = Math.abs(actual - expected) / Math.abs(expected);
        assertTrue(what + " : expected " + expected + " but was " + actual + " (relative " + err + ")",
                err <= tol);
    }

    private static void assertRelF(String what, float expected, float actual, float tol) {
        float err = Math.abs(actual - expected) / Math.abs(expected);
        assertTrue(what + " : expected " + expected + " but was " + actual + " (relative " + err + ")",
                err <= tol);
    }
}
