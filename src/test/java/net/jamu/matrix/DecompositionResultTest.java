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

import net.jamu.complex.Zd;
import net.jamu.complex.Zf;

/**
 * The factors a decomposition returns have to reproduce the matrix they came
 * from. The complex LU did not - its L and its P carried 1 + 1i where a one
 * belongs - so P L U was off by more than the matrix itself.
 */
public final class DecompositionResultTest {

    private static final long SEED = 5L;

    /** quadratic, tall and wide; the wide one is where getP() returns null */
    private static final int[][] SHAPES = { { 5, 5 }, { 6, 4 }, { 4, 6 }, { 12, 12 }, { 40, 40 } };

    /** measured worst is 4.3e-16 in double and 2.5e-7 in float */
    private static final double TOL_D = 1.0e-13;
    private static final float TOL_F = 1.0e-5f;

    private static double relD(MatrixD back, MatrixD a) {
        return back.addInplace(-1.0, a).normF() / a.normF();
    }

    private static double relCD(ComplexMatrixD back, ComplexMatrixD a) {
        return back.addInplace(-1.0, 0.0, a).normF() / a.normF();
    }

    private static float relF(MatrixF back, MatrixF a) {
        return back.addInplace(-1.0f, a).normF() / a.normF();
    }

    private static float relCF(ComplexMatrixF back, ComplexMatrixF a) {
        return back.addInplace(-1.0f, 0.0f, a).normF() / a.normF();
    }

    private static String at(int m, int n) {
        return " at " + m + "x" + n;
    }

    // ---------- P L U = A ----------

    @Test
    public void testTheLuFactorsReproduceTheMatrix() {
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            MatrixD a = Matrices.randomUniformD(m, n, SEED);
            LudD lu = a.lud();
            MatrixD back = lu.getPL().mult(lu.getU(), Matrices.createD(m, n));
            // relD subtracts in place, so it may be asked exactly once
            double e = relD(back, a);
            assertTrue("MatrixD P L U" + at(m, n) + ": " + e, e <= TOL_D);

            ComplexMatrixD c = Matrices.randomUniformComplexD(m, n, SEED);
            LudComplexD cl = c.lud();
            ComplexMatrixD cback = cl.getPL().mult(cl.getU(), Matrices.createComplexD(m, n));
            double ce = relCD(cback, c);
            assertTrue("ComplexMatrixD P L U" + at(m, n) + ": " + ce, ce <= TOL_D);

            MatrixF f = Matrices.randomUniformF(m, n, SEED);
            LudF fl = f.lud();
            MatrixF fback = fl.getPL().mult(fl.getU(), Matrices.createF(m, n));
            float fe = relF(fback, f);
            assertTrue("MatrixF P L U" + at(m, n) + ": " + fe, fe <= TOL_F);

            ComplexMatrixF g = Matrices.randomUniformComplexF(m, n, SEED);
            LudComplexF gl = g.lud();
            ComplexMatrixF gback = gl.getPL().mult(gl.getU(), Matrices.createComplexF(m, n));
            float ge = relCF(gback, g);
            assertTrue("ComplexMatrixF P L U" + at(m, n) + ": " + ge, ge <= TOL_F);
        }
    }

    @Test
    public void testTheUnitDiagonalOfLIsOne() {
        // exact, and it is what the defect got wrong: a unit lower triangular
        // factor whose diagonal is 1 + 1i is not one
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            MatrixD L = Matrices.randomUniformD(m, n, SEED).lud().getL();
            ComplexMatrixD cL = Matrices.randomUniformComplexD(m, n, SEED).lud().getL();
            MatrixF fL = Matrices.randomUniformF(m, n, SEED).lud().getL();
            ComplexMatrixF gL = Matrices.randomUniformComplexF(m, n, SEED).lud().getL();
            int k = Math.min(L.numRows(), L.numColumns());
            for (int i = 0; i < k; ++i) {
                assertEquals("MatrixD L" + at(m, n), 1.0, L.get(i, i), 0.0);
                Zd z = cL.getUnsafe(i, i);
                assertEquals("ComplexMatrixD L re" + at(m, n), Double.doubleToLongBits(1.0),
                        Double.doubleToLongBits(z.re()));
                assertEquals("ComplexMatrixD L im" + at(m, n), Double.doubleToLongBits(0.0),
                        Double.doubleToLongBits(z.im()));
                assertEquals("MatrixF L" + at(m, n), 1.0f, fL.get(i, i), 0.0f);
                Zf w = gL.getUnsafe(i, i);
                assertEquals("ComplexMatrixF L re" + at(m, n), Float.floatToIntBits(1.0f),
                        Float.floatToIntBits(w.re()));
                assertEquals("ComplexMatrixF L im" + at(m, n), Float.floatToIntBits(0.0f),
                        Float.floatToIntBits(w.im()));
            }
        }
    }

    @Test
    public void testThePermutationMatrixIsAPermutation() {
        // exact again: every entry is a zero or a one, and P is unitary
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            ComplexMatrixD p = Matrices.randomUniformComplexD(m, n, SEED).lud().getP();
            if (p == null) {
                // no row interchanges were needed, which is a documented answer
                continue;
            }
            int dim = p.numRows();
            for (int r = 0; r < dim; ++r) {
                for (int c = 0; c < dim; ++c) {
                    Zd z = p.getUnsafe(r, c);
                    assertEquals("P has an imaginary part" + at(m, n),
                            Double.doubleToLongBits(0.0), Double.doubleToLongBits(z.im()));
                    assertTrue("P entry is neither zero nor one" + at(m, n),
                            z.re() == 0.0 || z.re() == 1.0);
                }
            }
            ComplexMatrixD php = p.conjTransAmult(p, Matrices.createComplexD(dim, dim));
            ComplexMatrixD id = Matrices.identityComplexD(dim);
            assertEquals("P is not unitary" + at(m, n), 0.0, relCD(php, id), 0.0);

            ComplexMatrixF q = Matrices.randomUniformComplexF(m, n, SEED).lud().getP();
            int fdim = q.numRows();
            ComplexMatrixF qhq = q.conjTransAmult(q, Matrices.createComplexF(fdim, fdim));
            assertEquals("the float P is not unitary" + at(m, n), 0.0f,
                    relCF(qhq, Matrices.identityComplexF(fdim)), 0.0f);

            MatrixD rp = Matrices.randomUniformD(m, n, SEED).lud().getP();
            if (rp != null) {
                MatrixD ptp = rp.transAmult(rp, Matrices.createD(dim, dim));
                assertEquals("the real P is not orthogonal" + at(m, n), 0.0,
                        relD(ptp, Matrices.identityD(dim)), 0.0);
            }
        }
    }

    @Test
    public void testTheTailOfThePermutationIsFilledToo() {
        // P is as wide as L is tall, but the pivot vector is only as long as
        // the shorter side. Where the pivots all stay in the top rows, the
        // remaining diagonal of P comes from a branch of its own, and a 6 x 4
        // matrix whose last two rows are tiny is what reaches it
        int m = 6;
        int n = 4;
        ComplexMatrixD a = Matrices.randomUniformComplexD(m, n, SEED);
        for (int i = n; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                Zd z = a.getUnsafe(i, j);
                a.set(i, j, 1.0e-6 * z.re(), 1.0e-6 * z.im());
            }
        }
        LudComplexD lu = a.lud();
        ComplexMatrixD p = lu.getP();
        assertEquals("P should be as tall as L", m, p.numRows());
        for (int i = n; i < m; ++i) {
            Zd z = p.getUnsafe(i, i);
            assertEquals("the tail of P re", Double.doubleToLongBits(1.0),
                    Double.doubleToLongBits(z.re()));
            assertEquals("the tail of P im", Double.doubleToLongBits(0.0),
                    Double.doubleToLongBits(z.im()));
        }
        ComplexMatrixD php = p.conjTransAmult(p, Matrices.createComplexD(m, m));
        assertEquals("P is not unitary", 0.0, relCD(php, Matrices.identityComplexD(m)), 0.0);
        ComplexMatrixD back = lu.getPL().mult(lu.getU(), Matrices.createComplexD(m, n));
        double e = relCD(back, a);
        assertTrue("P L U with a filled tail: " + e, e <= TOL_D);
    }

    @Test
    public void testASingularMatrixIsStillFactored() {
        // the rank one matrix a[i][j] = (i+1)(j+1), where getrf reports a zero
        // pivot and the factors are nevertheless the right ones
        int n = 4;
        MatrixD a = Matrices.createD(n, n);
        ComplexMatrixD c = Matrices.createComplexD(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                a.set(i, j, (i + 1.0) * (j + 1.0));
                c.set(i, j, (i + 1.0) * (j + 1.0), 0.0);
            }
        }
        LudD lu = a.lud();
        assertTrue("the real matrix is singular", lu.isSingular());
        MatrixD back = lu.getPL().mult(lu.getU(), Matrices.createD(n, n));
        assertEquals("MatrixD P L U of a singular matrix", 0.0, relD(back, a), 0.0);

        LudComplexD cl = c.lud();
        assertTrue("the complex matrix is singular", cl.isSingular());
        ComplexMatrixD cback = cl.getPL().mult(cl.getU(), Matrices.createComplexD(n, n));
        assertEquals("ComplexMatrixD P L U of a singular matrix", 0.0, relCD(cback, c), 0.0);
    }

    // ---------- the rest of the family, which was already right ----------

    @Test
    public void testTheQrFactorsReproduceTheMatrix() {
        for (int[] s : SHAPES) {
            int m = s[0];
            int n = s[1];
            if (m < n) {
                // qrd wants m >= n and says so
                continue;
            }
            MatrixD a = Matrices.randomUniformD(m, n, SEED);
            QrdD q = a.qrd();
            MatrixD back = q.getQ().mult(q.getR(), Matrices.createD(m, n));
            assertTrue("MatrixD Q R" + at(m, n), relD(back, a) <= TOL_D);

            ComplexMatrixD c = Matrices.randomUniformComplexD(m, n, SEED);
            QrdComplexD cq = c.qrd();
            ComplexMatrixD cback = cq.getQ().mult(cq.getR(), Matrices.createComplexD(m, n));
            assertTrue("ComplexMatrixD Q R" + at(m, n), relCD(cback, c) <= TOL_D);

            MatrixF f = Matrices.randomUniformF(m, n, SEED);
            QrdF fq = f.qrd();
            MatrixF fback = fq.getQ().mult(fq.getR(), Matrices.createF(m, n));
            assertTrue("MatrixF Q R" + at(m, n), relF(fback, f) <= TOL_F);

            ComplexMatrixF g = Matrices.randomUniformComplexF(m, n, SEED);
            QrdComplexF gq = g.qrd();
            ComplexMatrixF gback = gq.getQ().mult(gq.getR(), Matrices.createComplexF(m, n));
            assertTrue("ComplexMatrixF Q R" + at(m, n), relCF(gback, g) <= TOL_F);
        }
    }

    @Test
    public void testTheEigenvectorsSolveTheEigenproblem() {
        // A V = V diag(lambda), for the complex side, where the eigenvectors
        // come back as complex columns; the real hierarchy packs a conjugate
        // pair into two real columns and is left out here
        for (int n : new int[] { 3, 10, 30 }) {
            ComplexMatrixD c = Matrices.randomUniformComplexD(n, n, SEED);
            EvdComplexD e = c.evd(true);
            ComplexMatrixD V = e.getEigenvectors();
            ComplexMatrixD av = c.mult(V, Matrices.createComplexD(n, n));
            ComplexMatrixD vl = V.mult(Matrices.diagComplexD(e.getEigenvalues()),
                    Matrices.createComplexD(n, n));
            assertTrue("A V against V diag(lambda) at " + n, relCD(av, vl) <= 1.0e-12);

            ComplexMatrixF g = Matrices.randomUniformComplexF(n, n, SEED);
            EvdComplexF ef = g.evd(true);
            ComplexMatrixF W = ef.getEigenvectors();
            ComplexMatrixF aw = g.mult(W, Matrices.createComplexF(n, n));
            ComplexMatrixF wl = W.mult(Matrices.diagComplexF(ef.getEigenvalues()),
                    Matrices.createComplexF(n, n));
            assertTrue("the float A V against V diag(lambda) at " + n, relCF(aw, wl) <= TOL_F);
        }
    }
}
