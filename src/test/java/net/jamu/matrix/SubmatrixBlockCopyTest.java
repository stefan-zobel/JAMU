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
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Random;

import org.junit.Test;

/**
 * Pins the behaviour of {@code submatrix} and {@code setSubmatrixInplace} in all
 * four matrix hierarchies. A rectangular block must land bit for bit at the
 * requested position, every cell outside that block must stay untouched, and
 * every out of range index combination must still be rejected.
 * <p>
 * These methods perform no arithmetic, so the comparison is on raw bits rather
 * than on a tolerance: {@code assertArrayEquals} with a zero delta would let
 * {@code -0.0} pass for {@code 0.0}.
 * <p>
 * The expected values come from a naive scalar loop written out in this class,
 * not from the library, so that a defect in the library cannot excuse itself.
 * Nothing here touches BLAS or LAPACK, so the test runs without a native MKL.
 */
public final class SubmatrixBlockCopyTest {

    private static final long SEED = 20260909L;

    /** source shapes for the exhaustive sweep */
    private static final int[][] SMALL = { { 1, 1 }, { 1, 4 }, { 4, 1 }, { 2, 3 }, { 3, 2 }, { 5, 5 } };

    /** target shapes for the exhaustive sweep */
    private static final int[][] SMALL_TARGETS = { { 5, 5 }, { 6, 7 }, { 1, 1 } };

    /** shapes for the seeded random sweep, big enough to cross any threshold */
    private static final int[][] LARGE = { { 40, 25 }, { 25, 40 }, { 64, 64 }, { 48, 3 }, { 3, 48 } };

    // ---------------------------------------------------------------- tests

    @Test
    public void testEveryBlockOfEverySmallShapeIsCopiedBitExact() {
        int checked = 0;
        for (int[] s : SMALL) {
            for (int[] t : SMALL_TARGETS) {
                checked += sweepExhaustively(s[0], s[1], t[0], t[1]);
            }
        }
        // a guard against a loop that silently degenerates to nothing
        assertTrue("the exhaustive sweep must actually run, was " + checked, checked > 5000);
    }

    @Test
    public void testTheBlockHeightsAroundTheArraycopyThresholdAreCopiedBitExact() {
        // the implementation may switch from an element loop to a block move at some
        // block height, so every height on both sides of any such threshold is covered
        int rows = 48;
        int cols = 48;
        Fixture f = new Fixture(rows, cols, rows, cols);
        for (int len = 1; len <= rows; ++len) {
            for (int width : new int[] { 1, 2, 3, 17, 48 }) {
                // once flush with the upper left corner, once shifted into the interior
                f.check("len " + len + " width " + width + " at (0,0)", 0, 0, len - 1, width - 1, 0, 0);
                f.check("len " + len + " width " + width + " shifted", 0, 0, len - 1, width - 1,
                        Math.min(3, rows - len), Math.min(5, cols - width));
            }
        }
    }

    @Test
    public void testBlocksBetweenMatricesOfDifferentHeightAreCopiedBitExact() {
        // when source and target have the same number of rows the two column
        // strides coincide and a mixed up stride goes unnoticed, so every pair
        // here has a different height, a non zero block origin and a non zero
        // target position
        int[] heights = { 41, 55, 64, 96 };
        for (int sr : heights) {
            for (int dr : heights) {
                if (sr == dr) {
                    continue;
                }
                Fixture f = new Fixture(sr, 9, dr, 11);
                int maxLen = Math.min(sr, dr);
                for (int len : new int[] { 1, 39, 40, 41, maxLen }) {
                    for (int rb0 : new int[] { 0, 1 }) {
                        if (len > maxLen || rb0 + len > sr) {
                            continue;
                        }
                        for (int cb0 : new int[] { 0, 3 }) {
                            for (int c0 : new int[] { 0, 2 }) {
                                int w = Math.min(9 - cb0, 11 - c0);
                                f.check("src " + sr + "x9 -> target " + dr + "x11, len " + len + ", rb0 "
                                        + rb0, rb0, cb0, rb0 + len - 1, cb0 + w - 1, dr - len, c0);
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    public void testRandomBlocksInLargeMatricesAreCopiedBitExact() {
        Random rnd = new Random(SEED);
        for (int[] s : LARGE) {
            for (int[] t : LARGE) {
                Fixture f = new Fixture(s[0], s[1], t[0], t[1]);
                for (int draw = 0; draw < 60; ++draw) {
                    int rb0 = rnd.nextInt(s[0]);
                    int rb1 = rb0 + rnd.nextInt(s[0] - rb0);
                    int cb0 = rnd.nextInt(s[1]);
                    int cb1 = cb0 + rnd.nextInt(s[1] - cb0);
                    int h = rb1 - rb0 + 1;
                    int w = cb1 - cb0 + 1;
                    if (h > t[0] || w > t[1]) {
                        continue;
                    }
                    f.check("random draw " + draw, rb0, cb0, rb1, cb1, rnd.nextInt(t[0] - h + 1),
                            rnd.nextInt(t[1] - w + 1));
                }
            }
        }
    }

    @Test
    public void testSpecialValuesSurviveTheCopyBitExact() {
        double[] specialD = { 0.0, -0.0, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                Double.MIN_VALUE, Double.MAX_VALUE, -1.0, 1.0, -Double.MIN_NORMAL, 0.0, -0.0 };
        float[] specialF = { 0.0f, -0.0f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY,
                Float.MIN_VALUE, Float.MAX_VALUE, -1.0f, 1.0f, -Float.MIN_NORMAL, 0.0f, -0.0f };
        int n = specialD.length;

        MatrixD sd = Matrices.createD(n, 2);
        MatrixF sf = Matrices.createF(n, 2);
        ComplexMatrixD scd = Matrices.createComplexD(n, 2);
        ComplexMatrixF scf = Matrices.createComplexF(n, 2);
        for (int col = 0; col < 2; ++col) {
            for (int row = 0; row < n; ++row) {
                int k = (row + col) % n;
                int k2 = (row + col + 1) % n;
                sd.set(row, col, specialD[k]);
                sf.set(row, col, specialF[k]);
                scd.set(row, col, specialD[k], specialD[k2]);
                scf.set(row, col, specialF[k], specialF[k2]);
            }
        }
        MatrixD dd = Matrices.randomUniformD(n, 2, SEED);
        MatrixF df = Matrices.randomUniformF(n, 2, SEED);
        ComplexMatrixD dcd = Matrices.randomUniformComplexD(n, 2, SEED);
        ComplexMatrixF dcf = Matrices.randomUniformComplexF(n, 2, SEED);
        double[] pd = dd.getArrayUnsafe().clone();
        float[] pf = df.getArrayUnsafe().clone();
        double[] pcd = dcd.getArrayUnsafe().clone();
        float[] pcf = dcf.getArrayUnsafe().clone();

        // a block height on either side of any threshold must see all of them
        for (int len = 1; len <= n; ++len) {
            String what = "special values, len " + len;
            setSubD(what + " MatrixD", pd, dd, sd, 1, 0, 0, 0, 0, len - 1, 1);
            setSubF(what + " MatrixF", pf, df, sf, 1, 0, 0, 0, 0, len - 1, 1);
            setSubCD(what + " ComplexMatrixD", pcd, dcd, scd, 0, 0, 0, 0, len - 1, 1);
            setSubCF(what + " ComplexMatrixF", pcf, dcf, scf, 0, 0, 0, 0, len - 1, 1);
        }
    }

    @Test
    public void testSetColumnInplaceCopiesTheWholeColumnBitExact() {
        for (int rows : new int[] { 1, 3, 16, 17, 64 }) {
            String at = " at " + rows + " rows";
            MatrixD d = Matrices.randomUniformD(rows, 4, SEED);
            MatrixD colD = Matrices.randomUniformD(rows, 1, SEED + 1L);
            double[] wantD = d.getArrayUnsafe().clone();
            refBlockD(colD.getArrayUnsafe(), rows, 0, 0, rows - 1, 0, wantD, rows, 0, 2, 1);
            d.setColumnInplace(2, colD);
            assertBitsD("MatrixD setColumnInplace" + at, wantD, d.getArrayUnsafe());

            MatrixF f = Matrices.randomUniformF(rows, 4, SEED);
            MatrixF colF = Matrices.randomUniformF(rows, 1, SEED + 1L);
            float[] wantF = f.getArrayUnsafe().clone();
            refBlockF(colF.getArrayUnsafe(), rows, 0, 0, rows - 1, 0, wantF, rows, 0, 2, 1);
            f.setColumnInplace(2, colF);
            assertBitsF("MatrixF setColumnInplace" + at, wantF, f.getArrayUnsafe());

            ComplexMatrixD cd = Matrices.randomUniformComplexD(rows, 4, SEED);
            ComplexMatrixD colCD = Matrices.randomUniformComplexD(rows, 1, SEED + 1L);
            double[] wantCD = cd.getArrayUnsafe().clone();
            refBlockD(colCD.getArrayUnsafe(), rows, 0, 0, rows - 1, 0, wantCD, rows, 0, 2, 2);
            cd.setColumnInplace(2, colCD);
            assertBitsD("ComplexMatrixD setColumnInplace" + at, wantCD, cd.getArrayUnsafe());

            ComplexMatrixF cf = Matrices.randomUniformComplexF(rows, 4, SEED);
            ComplexMatrixF colCF = Matrices.randomUniformComplexF(rows, 1, SEED + 1L);
            float[] wantCF = cf.getArrayUnsafe().clone();
            refBlockF(colCF.getArrayUnsafe(), rows, 0, 0, rows - 1, 0, wantCF, rows, 0, 2, 2);
            cf.setColumnInplace(2, colCF);
            assertBitsF("ComplexMatrixF setColumnInplace" + at, wantCF, cf.getArrayUnsafe());
        }
    }

    @Test
    public void testTheReturnedReferenceIsTheDocumentedOne() {
        MatrixD a = Matrices.randomUniformD(20, 20, SEED);
        MatrixD b = Matrices.randomUniformD(20, 20, SEED + 1L);
        assertSame("setSubmatrixInplace returns this", a, a.setSubmatrixInplace(0, 0, b, 0, 0, 19, 19));
        assertSame("submatrix returns B", b, a.submatrix(0, 0, 19, 19, b, 0, 0));
        ComplexMatrixD ca = Matrices.randomUniformComplexD(20, 20, SEED);
        ComplexMatrixD cb = Matrices.randomUniformComplexD(20, 20, SEED + 1L);
        assertSame("complex setSubmatrixInplace returns this", ca,
                ca.setSubmatrixInplace(0, 0, cb, 0, 0, 19, 19));
        assertSame("complex submatrix returns B", cb, ca.submatrix(0, 0, 19, 19, cb, 0, 0));
    }

    @Test
    public void testIllegalIndexesAreStillRejected() {
        MatrixD a = Matrices.randomUniformD(20, 20, SEED);
        MatrixD small = Matrices.randomUniformD(4, 4, SEED);
        rejects("negative source row", () -> a.setSubmatrixInplace(0, 0, small, -1, 0, 2, 2));
        rejects("negative source col", () -> a.setSubmatrixInplace(0, 0, small, 0, -1, 2, 2));
        rejects("source row past the end", () -> a.setSubmatrixInplace(0, 0, small, 0, 0, 4, 2));
        rejects("source col past the end", () -> a.setSubmatrixInplace(0, 0, small, 0, 0, 2, 4));
        rejects("inverted source rows", () -> a.setSubmatrixInplace(0, 0, small, 2, 0, 1, 2));
        rejects("inverted source cols", () -> a.setSubmatrixInplace(0, 0, small, 0, 2, 2, 1));
        rejects("negative target row", () -> a.setSubmatrixInplace(-1, 0, small, 0, 0, 2, 2));
        rejects("negative target col", () -> a.setSubmatrixInplace(0, -1, small, 0, 0, 2, 2));
        rejects("block overruns the target rows", () -> small.setSubmatrixInplace(2, 0, a, 0, 0, 3, 3));
        rejects("block overruns the target cols", () -> small.setSubmatrixInplace(0, 2, a, 0, 0, 3, 3));

        rejects("submatrix: negative row", () -> a.submatrix(-1, 0, 2, 2, small, 0, 0));
        rejects("submatrix: row past the end", () -> a.submatrix(0, 0, 20, 2, small, 0, 0));
        rejects("submatrix: inverted rows", () -> a.submatrix(2, 0, 1, 2, small, 0, 0));
        rejects("submatrix: negative target row", () -> a.submatrix(0, 0, 2, 2, small, -1, 0));
        rejects("submatrix: overruns the target", () -> a.submatrix(0, 0, 5, 5, small, 0, 0));

        ComplexMatrixD ca = Matrices.randomUniformComplexD(20, 20, SEED);
        ComplexMatrixD csmall = Matrices.randomUniformComplexD(4, 4, SEED);
        rejects("complex: negative source row", () -> ca.setSubmatrixInplace(0, 0, csmall, -1, 0, 2, 2));
        rejects("complex: source row past the end", () -> ca.setSubmatrixInplace(0, 0, csmall, 0, 0, 4, 2));
        rejects("complex: overruns the target", () -> csmall.setSubmatrixInplace(2, 0, ca, 0, 0, 3, 3));
        rejects("complex submatrix: row past the end", () -> ca.submatrix(0, 0, 20, 2, csmall, 0, 0));
        rejects("complex submatrix: overruns the target", () -> ca.submatrix(0, 0, 5, 5, csmall, 0, 0));
    }

    // ------------------------------------------------------------- drivers

    /**
     * Every block of a {@code sr x sc} source into every position of a
     * {@code dr x dc} target that can hold it, in both directions. Returns the
     * number of index combinations that were checked.
     */
    private static int sweepExhaustively(int sr, int sc, int dr, int dc) {
        Fixture f = new Fixture(sr, sc, dr, dc);
        String what = sr + "x" + sc + " -> " + dr + "x" + dc;
        int checked = 0;
        for (int rb0 = 0; rb0 < sr; ++rb0) {
            for (int rb1 = rb0; rb1 < sr; ++rb1) {
                int h = rb1 - rb0 + 1;
                if (h > dr) {
                    continue;
                }
                for (int cb0 = 0; cb0 < sc; ++cb0) {
                    for (int cb1 = cb0; cb1 < sc; ++cb1) {
                        int w = cb1 - cb0 + 1;
                        if (w > dc) {
                            continue;
                        }
                        for (int r0 = 0; r0 + h <= dr; ++r0) {
                            for (int c0 = 0; c0 + w <= dc; ++c0) {
                                f.check(what, rb0, cb0, rb1, cb1, r0, c0);
                                ++checked;
                            }
                        }
                    }
                }
            }
        }
        return checked;
    }

    /**
     * One source and one target of each of the four matrix types, reused across
     * many index combinations. The target is restored from its pristine copy
     * before every single check, so the checks stay independent of each other.
     */
    private static final class Fixture {

        private final MatrixD sd;
        private final MatrixD dd;
        private final double[] pd;
        private final MatrixF sf;
        private final MatrixF df;
        private final float[] pf;
        private final ComplexMatrixD scd;
        private final ComplexMatrixD dcd;
        private final double[] pcd;
        private final ComplexMatrixF scf;
        private final ComplexMatrixF dcf;
        private final float[] pcf;

        Fixture(int sr, int sc, int dr, int dc) {
            sd = Matrices.randomUniformD(sr, sc, SEED);
            dd = Matrices.randomUniformD(dr, dc, SEED + 1L);
            pd = dd.getArrayUnsafe().clone();
            sf = Matrices.randomUniformF(sr, sc, SEED);
            df = Matrices.randomUniformF(dr, dc, SEED + 1L);
            pf = df.getArrayUnsafe().clone();
            scd = Matrices.randomUniformComplexD(sr, sc, SEED);
            dcd = Matrices.randomUniformComplexD(dr, dc, SEED + 1L);
            pcd = dcd.getArrayUnsafe().clone();
            scf = Matrices.randomUniformComplexF(sr, sc, SEED);
            dcf = Matrices.randomUniformComplexF(dr, dc, SEED + 1L);
            pcf = dcf.getArrayUnsafe().clone();
        }

        /**
         * Checks {@code setSubmatrixInplace} (the source block written into the
         * target) and {@code submatrix} (the same block read out of the source into
         * the target) for all four matrix types.
         */
        void check(String what, int rb0, int cb0, int rb1, int cb1, int r0, int c0) {
            setSubD(what + " MatrixD", pd, dd, sd, 1, r0, c0, rb0, cb0, rb1, cb1);
            subD(what + " MatrixD submatrix", pd, dd, sd, 1, rb0, cb0, rb1, cb1, r0, c0);
            setSubF(what + " MatrixF", pf, df, sf, 1, r0, c0, rb0, cb0, rb1, cb1);
            subF(what + " MatrixF submatrix", pf, df, sf, 1, rb0, cb0, rb1, cb1, r0, c0);
            setSubCD(what + " ComplexMatrixD", pcd, dcd, scd, r0, c0, rb0, cb0, rb1, cb1);
            subCD(what + " ComplexMatrixD submatrix", pcd, dcd, scd, rb0, cb0, rb1, cb1, r0, c0);
            setSubCF(what + " ComplexMatrixF", pcf, dcf, scf, r0, c0, rb0, cb0, rb1, cb1);
            subCF(what + " ComplexMatrixF submatrix", pcf, dcf, scf, rb0, cb0, rb1, cb1, r0, c0);
        }
    }

    // ------------------------------------------------- per type check bodies

    static void setSubD(String what, double[] pristine, MatrixD dst, MatrixD src, int e, int r0,
            int c0, int rb0, int cb0, int rb1, int cb1) {
        double[] want = restoreD(pristine, dst.getArrayUnsafe());
        refBlockD(src.getArrayUnsafe(), src.numRows(), rb0, cb0, rb1, cb1, want, dst.numRows(), r0, c0, e);
        dst.setSubmatrixInplace(r0, c0, src, rb0, cb0, rb1, cb1);
        assertBitsD(what, want, dst.getArrayUnsafe());
    }

    static void subD(String what, double[] pristine, MatrixD dst, MatrixD src, int e, int r0, int c0,
            int r1, int c1, int rb, int cb) {
        double[] want = restoreD(pristine, dst.getArrayUnsafe());
        refBlockD(src.getArrayUnsafe(), src.numRows(), r0, c0, r1, c1, want, dst.numRows(), rb, cb, e);
        src.submatrix(r0, c0, r1, c1, dst, rb, cb);
        assertBitsD(what, want, dst.getArrayUnsafe());
    }

    static void setSubF(String what, float[] pristine, MatrixF dst, MatrixF src, int e, int r0,
            int c0, int rb0, int cb0, int rb1, int cb1) {
        float[] want = restoreF(pristine, dst.getArrayUnsafe());
        refBlockF(src.getArrayUnsafe(), src.numRows(), rb0, cb0, rb1, cb1, want, dst.numRows(), r0, c0, e);
        dst.setSubmatrixInplace(r0, c0, src, rb0, cb0, rb1, cb1);
        assertBitsF(what, want, dst.getArrayUnsafe());
    }

    static void subF(String what, float[] pristine, MatrixF dst, MatrixF src, int e, int r0, int c0,
            int r1, int c1, int rb, int cb) {
        float[] want = restoreF(pristine, dst.getArrayUnsafe());
        refBlockF(src.getArrayUnsafe(), src.numRows(), r0, c0, r1, c1, want, dst.numRows(), rb, cb, e);
        src.submatrix(r0, c0, r1, c1, dst, rb, cb);
        assertBitsF(what, want, dst.getArrayUnsafe());
    }

    static void setSubCD(String what, double[] pristine, ComplexMatrixD dst, ComplexMatrixD src,
            int r0, int c0, int rb0, int cb0, int rb1, int cb1) {
        double[] want = restoreD(pristine, dst.getArrayUnsafe());
        refBlockD(src.getArrayUnsafe(), src.numRows(), rb0, cb0, rb1, cb1, want, dst.numRows(), r0, c0, 2);
        dst.setSubmatrixInplace(r0, c0, src, rb0, cb0, rb1, cb1);
        assertBitsD(what, want, dst.getArrayUnsafe());
    }

    static void subCD(String what, double[] pristine, ComplexMatrixD dst, ComplexMatrixD src, int r0,
            int c0, int r1, int c1, int rb, int cb) {
        double[] want = restoreD(pristine, dst.getArrayUnsafe());
        refBlockD(src.getArrayUnsafe(), src.numRows(), r0, c0, r1, c1, want, dst.numRows(), rb, cb, 2);
        src.submatrix(r0, c0, r1, c1, dst, rb, cb);
        assertBitsD(what, want, dst.getArrayUnsafe());
    }

    static void setSubCF(String what, float[] pristine, ComplexMatrixF dst, ComplexMatrixF src,
            int r0, int c0, int rb0, int cb0, int rb1, int cb1) {
        float[] want = restoreF(pristine, dst.getArrayUnsafe());
        refBlockF(src.getArrayUnsafe(), src.numRows(), rb0, cb0, rb1, cb1, want, dst.numRows(), r0, c0, 2);
        dst.setSubmatrixInplace(r0, c0, src, rb0, cb0, rb1, cb1);
        assertBitsF(what, want, dst.getArrayUnsafe());
    }

    static void subCF(String what, float[] pristine, ComplexMatrixF dst, ComplexMatrixF src, int r0,
            int c0, int r1, int c1, int rb, int cb) {
        float[] want = restoreF(pristine, dst.getArrayUnsafe());
        refBlockF(src.getArrayUnsafe(), src.numRows(), r0, c0, r1, c1, want, dst.numRows(), rb, cb, 2);
        src.submatrix(r0, c0, r1, c1, dst, rb, cb);
        assertBitsF(what, want, dst.getArrayUnsafe());
    }

    // --------------------------------------------------- reference and asserts

    /**
     * The expected result, computed the slow and obvious way. Storage is column
     * major with {@code e} array slots per matrix cell ({@code e == 2} for the
     * complex types, whose real and imaginary part are adjacent).
     */
    private static void refBlockD(double[] src, int srcRows, int rb0, int cb0, int rb1, int cb1, double[] dst,
            int dstRows, int r0, int c0, int e) {
        for (int col = cb0; col <= cb1; ++col) {
            for (int row = rb0; row <= rb1; ++row) {
                int from = e * (col * srcRows + row);
                int to = e * ((c0 + col - cb0) * dstRows + (r0 + row - rb0));
                for (int k = 0; k < e; ++k) {
                    dst[to + k] = src[from + k];
                }
            }
        }
    }

    private static void refBlockF(float[] src, int srcRows, int rb0, int cb0, int rb1, int cb1, float[] dst,
            int dstRows, int r0, int c0, int e) {
        for (int col = cb0; col <= cb1; ++col) {
            for (int row = rb0; row <= rb1; ++row) {
                int from = e * (col * srcRows + row);
                int to = e * ((c0 + col - cb0) * dstRows + (r0 + row - rb0));
                for (int k = 0; k < e; ++k) {
                    dst[to + k] = src[from + k];
                }
            }
        }
    }

    /** puts the target back into its pristine state and returns a fresh expectation */
    private static double[] restoreD(double[] pristine, double[] target) {
        System.arraycopy(pristine, 0, target, 0, pristine.length);
        return pristine.clone();
    }

    private static float[] restoreF(float[] pristine, float[] target) {
        System.arraycopy(pristine, 0, target, 0, pristine.length);
        return pristine.clone();
    }

    private static void assertBitsD(String what, double[] expected, double[] actual) {
        assertEquals(what + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            long e = Double.doubleToRawLongBits(expected[i]);
            long a = Double.doubleToRawLongBits(actual[i]);
            if (e != a) {
                fail(what + " differs at " + i + " : expected " + expected[i] + " but was " + actual[i]);
            }
        }
    }

    private static void assertBitsF(String what, float[] expected, float[] actual) {
        assertEquals(what + " length", expected.length, actual.length);
        for (int i = 0; i < expected.length; ++i) {
            int e = Float.floatToRawIntBits(expected[i]);
            int a = Float.floatToRawIntBits(actual[i]);
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
            fail(what + " : expected an IllegalArgumentException but none was thrown");
        } catch (IllegalArgumentException expected) {
            // the guard did its job
        }
    }
}
