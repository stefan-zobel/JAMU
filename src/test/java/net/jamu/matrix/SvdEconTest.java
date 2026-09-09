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
 * Tests that pin the shapes the economy SVD returns.
 */
public final class SvdEconTest {

    private static final int[][] SHAPES = { { 6, 4 }, { 4, 6 }, { 5, 5 }, { 1, 3 }, { 3, 1 } };
    private static final double TOL_D = 1.0e-13;
    private static final float TOL_F = 1.0e-5f;

    @Test
    public void testTheShapesAreDrivenByMinOfRowsAndColumns() {
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            int k = Math.min(m, n);
            String at = " at " + m + "x" + n;

            SvdEconD d = Matrices.randomUniformD(m, n, 5L).svdEcon();
            assertShape("MatrixD U" + at, m, k, d.getU().numRows(), d.getU().numColumns());
            assertShape("MatrixD Vt" + at, k, n, d.getVt().numRows(), d.getVt().numColumns());
            assertEquals("MatrixD S" + at, k, d.getS().length);

            SvdEconF f = Matrices.randomUniformF(m, n, 5L).svdEcon();
            assertShape("MatrixF U" + at, m, k, f.getU().numRows(), f.getU().numColumns());
            assertShape("MatrixF Vt" + at, k, n, f.getVt().numRows(), f.getVt().numColumns());
            assertEquals("MatrixF S" + at, k, f.getS().length);

            SvdEconComplexD cd = Matrices.randomUniformComplexD(m, n, 5L).svdEcon();
            assertShape("ComplexMatrixD U" + at, m, k, cd.getU().numRows(), cd.getU().numColumns());
            assertShape("ComplexMatrixD Vh" + at, k, n, cd.getVh().numRows(), cd.getVh().numColumns());
            assertEquals("ComplexMatrixD S" + at, k, cd.getS().length);

            SvdEconComplexF cf = Matrices.randomUniformComplexF(m, n, 5L).svdEcon();
            assertShape("ComplexMatrixF U" + at, m, k, cf.getU().numRows(), cf.getU().numColumns());
            assertShape("ComplexMatrixF Vh" + at, k, n, cf.getVh().numRows(), cf.getVh().numColumns());
            assertEquals("ComplexMatrixF S" + at, k, cf.getS().length);
        }
    }

    @Test
    public void testARankDeficientMatrixIsNotTruncated() {
        MatrixD a = rankTwo();
        assertEquals("the fixture must be rank 2", 2, Matrices.numericalRank(a));
        SvdEconD d = a.svdEcon();
        assertShape("U", 6, 4, d.getU().numRows(), d.getU().numColumns());
        assertShape("Vt", 4, 4, d.getVt().numRows(), d.getVt().numColumns());
        assertEquals("S has min(m, n) entries, not the rank", 4, d.getS().length);
        double[] sv = d.getS();
        assertTrue("the two leading values are not small", sv[1] > 1.0e-3);
        assertTrue("the third value is zero to machine precision, was " + sv[2],
                sv[2] < 1.0e-13 * sv[0]);
        assertTrue("the fourth value is zero to machine precision, was " + sv[3],
                sv[3] < 1.0e-13 * sv[0]);
    }

    @Test
    public void testTheThinFactorsReproduceTheMatrix() {
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            int k = Math.min(m, n);
            String at = " at " + m + "x" + n;
            MatrixD a = Matrices.randomUniformD(m, n, 5L);
            SvdEconD d = a.svdEcon();
            MatrixD back = d.getU().times(Matrices.diagD(k, k, d.getS())).times(d.getVt());
            assertTrue("U S Vt does not reproduce A" + at, relD(back, a) <= TOL_D);

            MatrixF af = Matrices.randomUniformF(m, n, 5L);
            SvdEconF f = af.svdEcon();
            MatrixF backF = f.getU().times(Matrices.diagF(k, k, f.getS())).times(f.getVt());
            assertTrue("float U S Vt does not reproduce A" + at, relF(backF, af) <= TOL_F);
        }
        MatrixD a = rankTwo();
        SvdEconD d = a.svdEcon();
        MatrixD back = d.getU().times(Matrices.diagD(4, 4, d.getS())).times(d.getVt());
        assertTrue("U S Vt does not reproduce a rank deficient A", relD(back, a) <= TOL_D);
    }

    @Test
    public void testTheEconomyFormIsSmallerThanTheFullOne() {
        MatrixD a = Matrices.randomUniformD(6, 4, 5L);
        SvdD full = a.svd(true);
        assertShape("full U", 6, 6, full.getU().numRows(), full.getU().numColumns());
        assertShape("full Vt", 4, 4, full.getVt().numRows(), full.getVt().numColumns());
        SvdEconD econ = a.svdEcon();
        assertShape("economy U", 6, 4, econ.getU().numRows(), econ.getU().numColumns());
        assertShape("economy Vt", 4, 4, econ.getVt().numRows(), econ.getVt().numColumns());
        assertEquals("both report the same number of singular values", full.getS().length,
                econ.getS().length);
    }

    @Test
    public void testTheSingularVectorsAreAlwaysComputed() {
        assertTrue("MatrixD", Matrices.randomUniformD(6, 4, 5L).svdEcon().hasSingularVectors());
        assertTrue("MatrixF", Matrices.randomUniformF(6, 4, 5L).svdEcon().hasSingularVectors());
        assertTrue("ComplexMatrixD",
                Matrices.randomUniformComplexD(6, 4, 5L).svdEcon().hasSingularVectors());
        assertTrue("ComplexMatrixF",
                Matrices.randomUniformComplexF(6, 4, 5L).svdEcon().hasSingularVectors());
    }

    /** a 6 x 4 matrix of rank 2, built as a product of a 6 x 2 and a 2 x 4 */
    private static MatrixD rankTwo() {
        return Matrices.randomUniformD(6, 2, 7L).times(Matrices.randomUniformD(2, 4, 8L));
    }

    private static void assertShape(String what, int rows, int cols, int gotRows, int gotCols) {
        assertEquals(what + " rows", rows, gotRows);
        assertEquals(what + " columns", cols, gotCols);
    }

    private static double relD(MatrixD got, MatrixD want) {
        return got.copy().addInplace(-1.0, want).normF() / want.normF();
    }

    private static float relF(MatrixF got, MatrixF want) {
        return got.copy().addInplace(-1.0f, want).normF() / want.normF();
    }
}
