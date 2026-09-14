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

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.lang.reflect.Method;

import org.junit.Assume;
import org.junit.Test;

/**
 * Pins the copies that {@code solve}, its callers and {@code timesMany} make on a
 * view, and that {@code solve} and {@code inv} may write into their own matrix.
 */
public final class ViewCopyTest {

    private static final long SEED = 20260919L;

    /** dimension for the allocation checks */
    private static final int N = 300;

    // ---------------------------------------------------------------- results

    @Test
    public void testMatrixDViewSolvesLikeItsCopy() {
        MatrixD P = dominantD(9, SEED);
        MatrixD G = Matrices.randomUniformD(12, 11, SEED + 1);
        double[] p0 = P.getArrayUnsafe().clone();
        double[] g0 = G.getArrayUnsafe().clone();
        MatrixD sq = Matrices.view(P, 2, 2, 7, 7);
        MatrixD tall = Matrices.view(G, 1, 3, 9, 7);
        MatrixD wide = Matrices.view(G, 3, 0, 6, 8);
        MatrixD b6 = Matrices.randomUniformD(6, 2, SEED + 2);
        MatrixD b9 = Matrices.randomUniformD(9, 2, SEED + 3);
        MatrixD b4 = Matrices.randomUniformD(4, 2, SEED + 4);
        assertClose("solve square", sq.copy().solve(b6, Matrices.createD(6, 2)),
                sq.solve(b6, Matrices.createD(6, 2)));
        assertClose("solve tall", tall.copy().solve(b9, Matrices.createD(5, 2)),
                tall.solve(b9, Matrices.createD(5, 2)));
        assertClose("solve wide", wide.copy().solve(b4, Matrices.createD(9, 2)),
                wide.solve(b4, Matrices.createD(9, 2)));
        assertClose("inv", sq.copy().inv(Matrices.createD(6, 6)), sq.inv(Matrices.createD(6, 6)));
        assertClose("inverse", sq.copy().inverse(), sq.inverse());
        assertClose("mldivide square", sq.copy().mldivide(b6), sq.mldivide(b6));
        assertClose("mldivide tall", tall.copy().mldivide(b9), tall.mldivide(b9));
        assertBits("P", p0, P.getArrayUnsafe());
        assertBits("G", g0, G.getArrayUnsafe());
    }

    @Test
    public void testMatrixFViewSolvesLikeItsCopy() {
        MatrixF P = dominantF(9, SEED);
        MatrixF G = Matrices.randomUniformF(12, 11, SEED + 1);
        float[] p0 = P.getArrayUnsafe().clone();
        float[] g0 = G.getArrayUnsafe().clone();
        MatrixF sq = Matrices.view(P, 2, 2, 7, 7);
        MatrixF tall = Matrices.view(G, 1, 3, 9, 7);
        MatrixF wide = Matrices.view(G, 3, 0, 6, 8);
        MatrixF b6 = Matrices.randomUniformF(6, 2, SEED + 2);
        MatrixF b9 = Matrices.randomUniformF(9, 2, SEED + 3);
        MatrixF b4 = Matrices.randomUniformF(4, 2, SEED + 4);
        assertClose("solve square", sq.copy().solve(b6, Matrices.createF(6, 2)),
                sq.solve(b6, Matrices.createF(6, 2)));
        assertClose("solve tall", tall.copy().solve(b9, Matrices.createF(5, 2)),
                tall.solve(b9, Matrices.createF(5, 2)));
        assertClose("solve wide", wide.copy().solve(b4, Matrices.createF(9, 2)),
                wide.solve(b4, Matrices.createF(9, 2)));
        assertClose("inv", sq.copy().inv(Matrices.createF(6, 6)), sq.inv(Matrices.createF(6, 6)));
        assertClose("inverse", sq.copy().inverse(), sq.inverse());
        assertClose("mldivide square", sq.copy().mldivide(b6), sq.mldivide(b6));
        assertClose("mldivide tall", tall.copy().mldivide(b9), tall.mldivide(b9));
        assertBits("P", p0, P.getArrayUnsafe());
        assertBits("G", g0, G.getArrayUnsafe());
    }

    @Test
    public void testComplexMatrixDViewSolvesLikeItsCopy() {
        ComplexMatrixD P = dominantComplexD(9, SEED);
        ComplexMatrixD G = Matrices.randomUniformComplexD(12, 11, SEED + 1);
        double[] p0 = P.getArrayUnsafe().clone();
        double[] g0 = G.getArrayUnsafe().clone();
        ComplexMatrixD sq = Matrices.view(P, 2, 2, 7, 7);
        ComplexMatrixD tall = Matrices.view(G, 1, 3, 9, 7);
        ComplexMatrixD wide = Matrices.view(G, 3, 0, 6, 8);
        ComplexMatrixD b6 = Matrices.randomUniformComplexD(6, 2, SEED + 2);
        ComplexMatrixD b9 = Matrices.randomUniformComplexD(9, 2, SEED + 3);
        ComplexMatrixD b4 = Matrices.randomUniformComplexD(4, 2, SEED + 4);
        assertClose("solve square", sq.copy().solve(b6, Matrices.createComplexD(6, 2)),
                sq.solve(b6, Matrices.createComplexD(6, 2)));
        assertClose("solve tall", tall.copy().solve(b9, Matrices.createComplexD(5, 2)),
                tall.solve(b9, Matrices.createComplexD(5, 2)));
        assertClose("solve wide", wide.copy().solve(b4, Matrices.createComplexD(9, 2)),
                wide.solve(b4, Matrices.createComplexD(9, 2)));
        assertClose("inv", sq.copy().inv(Matrices.createComplexD(6, 6)), sq.inv(Matrices.createComplexD(6, 6)));
        assertClose("inverse", sq.copy().inverse(), sq.inverse());
        assertClose("mldivide square", sq.copy().mldivide(b6), sq.mldivide(b6));
        assertClose("mldivide tall", tall.copy().mldivide(b9), tall.mldivide(b9));
        assertBits("P", p0, P.getArrayUnsafe());
        assertBits("G", g0, G.getArrayUnsafe());
    }

    @Test
    public void testComplexMatrixFViewSolvesLikeItsCopy() {
        ComplexMatrixF P = dominantComplexF(9, SEED);
        ComplexMatrixF G = Matrices.randomUniformComplexF(12, 11, SEED + 1);
        float[] p0 = P.getArrayUnsafe().clone();
        float[] g0 = G.getArrayUnsafe().clone();
        ComplexMatrixF sq = Matrices.view(P, 2, 2, 7, 7);
        ComplexMatrixF tall = Matrices.view(G, 1, 3, 9, 7);
        ComplexMatrixF wide = Matrices.view(G, 3, 0, 6, 8);
        ComplexMatrixF b6 = Matrices.randomUniformComplexF(6, 2, SEED + 2);
        ComplexMatrixF b9 = Matrices.randomUniformComplexF(9, 2, SEED + 3);
        ComplexMatrixF b4 = Matrices.randomUniformComplexF(4, 2, SEED + 4);
        assertClose("solve square", sq.copy().solve(b6, Matrices.createComplexF(6, 2)),
                sq.solve(b6, Matrices.createComplexF(6, 2)));
        assertClose("solve tall", tall.copy().solve(b9, Matrices.createComplexF(5, 2)),
                tall.solve(b9, Matrices.createComplexF(5, 2)));
        assertClose("solve wide", wide.copy().solve(b4, Matrices.createComplexF(9, 2)),
                wide.solve(b4, Matrices.createComplexF(9, 2)));
        assertClose("inv", sq.copy().inv(Matrices.createComplexF(6, 6)), sq.inv(Matrices.createComplexF(6, 6)));
        assertClose("inverse", sq.copy().inverse(), sq.inverse());
        assertClose("mldivide square", sq.copy().mldivide(b6), sq.mldivide(b6));
        assertClose("mldivide tall", tall.copy().mldivide(b9), tall.mldivide(b9));
        assertBits("P", p0, P.getArrayUnsafe());
        assertBits("G", g0, G.getArrayUnsafe());
    }

    @Test
    public void testAViewSolvesIntoItsOwnParent() {
        MatrixD pd = dominantD(6, SEED + 5);
        MatrixD bd = Matrices.randomUniformD(6, 6, SEED + 6);
        MatrixD wd = pd.copy().solve(bd, Matrices.createD(6, 6));
        assertSame(pd, Matrices.view(pd, 0, 0, 5, 5).solve(bd, pd));
        assertClose("MatrixD solve", wd, pd);
        pd = dominantD(6, SEED + 5);
        wd = pd.copy().inverse();
        assertSame(pd, Matrices.view(pd, 0, 0, 5, 5).inv(pd));
        assertClose("MatrixD inv", wd, pd);

        MatrixF pf = dominantF(6, SEED + 5);
        MatrixF bf = Matrices.randomUniformF(6, 6, SEED + 6);
        MatrixF wf = pf.copy().solve(bf, Matrices.createF(6, 6));
        assertSame(pf, Matrices.view(pf, 0, 0, 5, 5).solve(bf, pf));
        assertClose("MatrixF solve", wf, pf);
        pf = dominantF(6, SEED + 5);
        wf = pf.copy().inverse();
        assertSame(pf, Matrices.view(pf, 0, 0, 5, 5).inv(pf));
        assertClose("MatrixF inv", wf, pf);

        ComplexMatrixD pzd = dominantComplexD(6, SEED + 5);
        ComplexMatrixD bzd = Matrices.randomUniformComplexD(6, 6, SEED + 6);
        ComplexMatrixD wzd = pzd.copy().solve(bzd, Matrices.createComplexD(6, 6));
        assertSame(pzd, Matrices.view(pzd, 0, 0, 5, 5).solve(bzd, pzd));
        assertClose("ComplexMatrixD solve", wzd, pzd);
        pzd = dominantComplexD(6, SEED + 5);
        wzd = pzd.copy().inverse();
        assertSame(pzd, Matrices.view(pzd, 0, 0, 5, 5).inv(pzd));
        assertClose("ComplexMatrixD inv", wzd, pzd);

        ComplexMatrixF pzf = dominantComplexF(6, SEED + 5);
        ComplexMatrixF bzf = Matrices.randomUniformComplexF(6, 6, SEED + 6);
        ComplexMatrixF wzf = pzf.copy().solve(bzf, Matrices.createComplexF(6, 6));
        assertSame(pzf, Matrices.view(pzf, 0, 0, 5, 5).solve(bzf, pzf));
        assertClose("ComplexMatrixF solve", wzf, pzf);
        pzf = dominantComplexF(6, SEED + 5);
        wzf = pzf.copy().inverse();
        assertSame(pzf, Matrices.view(pzf, 0, 0, 5, 5).inv(pzf));
        assertClose("ComplexMatrixF inv", wzf, pzf);
    }

    // ---------------------------------------------------------------- aliasing

    @Test
    public void testSolveAndInvMayWriteIntoTheirOwnMatrix() {
        MatrixD ad = dominantD(6, SEED + 7);
        MatrixD bd = Matrices.randomUniformD(6, 6, SEED + 8);
        MatrixD a0d = ad.copy();
        assertClose("MatrixD solve", a0d.solve(bd, Matrices.createD(6, 6)), ad.solve(bd, Matrices.createD(6, 6)));
        assertBits("MatrixD A", a0d.getArrayUnsafe(), ad.getArrayUnsafe());
        assertClose("MatrixD solve(B, A)", a0d.solve(bd, Matrices.createD(6, 6)), ad.solve(bd, ad));
        ad = a0d.copy();
        assertClose("MatrixD inv(A)", a0d.inverse(), ad.inv(ad));

        MatrixF af = dominantF(6, SEED + 7);
        MatrixF bf = Matrices.randomUniformF(6, 6, SEED + 8);
        MatrixF a0f = af.copy();
        assertClose("MatrixF solve", a0f.solve(bf, Matrices.createF(6, 6)), af.solve(bf, Matrices.createF(6, 6)));
        assertBits("MatrixF A", a0f.getArrayUnsafe(), af.getArrayUnsafe());
        assertClose("MatrixF solve(B, A)", a0f.solve(bf, Matrices.createF(6, 6)), af.solve(bf, af));
        af = a0f.copy();
        assertClose("MatrixF inv(A)", a0f.inverse(), af.inv(af));

        ComplexMatrixD azd = dominantComplexD(6, SEED + 7);
        ComplexMatrixD bzd = Matrices.randomUniformComplexD(6, 6, SEED + 8);
        ComplexMatrixD a0zd = azd.copy();
        assertClose("ComplexMatrixD solve", a0zd.solve(bzd, Matrices.createComplexD(6, 6)),
                azd.solve(bzd, Matrices.createComplexD(6, 6)));
        assertBits("ComplexMatrixD A", a0zd.getArrayUnsafe(), azd.getArrayUnsafe());
        assertClose("ComplexMatrixD solve(B, A)", a0zd.solve(bzd, Matrices.createComplexD(6, 6)),
                azd.solve(bzd, azd));
        azd = a0zd.copy();
        assertClose("ComplexMatrixD inv(A)", a0zd.inverse(), azd.inv(azd));

        ComplexMatrixF azf = dominantComplexF(6, SEED + 7);
        ComplexMatrixF bzf = Matrices.randomUniformComplexF(6, 6, SEED + 8);
        ComplexMatrixF a0zf = azf.copy();
        assertClose("ComplexMatrixF solve", a0zf.solve(bzf, Matrices.createComplexF(6, 6)),
                azf.solve(bzf, Matrices.createComplexF(6, 6)));
        assertBits("ComplexMatrixF A", a0zf.getArrayUnsafe(), azf.getArrayUnsafe());
        assertClose("ComplexMatrixF solve(B, A)", a0zf.solve(bzf, Matrices.createComplexF(6, 6)),
                azf.solve(bzf, azf));
        azf = a0zf.copy();
        assertClose("ComplexMatrixF inv(A)", a0zf.inverse(), azf.inv(azf));
    }

    // ---------------------------------------------------------------- allocations

    @Test
    public void testSolveOnAViewCopiesTheViewOnce() {
        Method bytes = allocatedBytes();
        Assume.assumeTrue("thread allocation counter not available", bytes != null);
        int h = N / 2;

        MatrixD sqd = Matrices.view(dominantD(N + 1, SEED), 1, 1, N, N);
        MatrixD tlld = Matrices.view(Matrices.randomUniformD(N + 1, h + 1, SEED), 1, 1, N, h);
        MatrixD bd = Matrices.randomUniformD(N, 1, SEED + 1);
        assertSaves(bytes, "MatrixD solve square", 8L * N * N, () -> sqd.solve(bd, Matrices.createD(N, 1)),
                () -> sqd.copy().solve(bd, Matrices.createD(N, 1)));
        assertSaves(bytes, "MatrixD solve tall", 8L * N * h, () -> tlld.solve(bd, Matrices.createD(h, 1)),
                () -> tlld.copy().solve(bd, Matrices.createD(h, 1)));
        assertSaves(bytes, "MatrixD inv", 8L * N * N, () -> sqd.inv(Matrices.createD(N, N)),
                () -> sqd.copy().inv(Matrices.createD(N, N)));
        assertSaves(bytes, "MatrixD inverse", 8L * N * N, () -> sqd.inverse(), () -> sqd.copy().inverse());
        assertSaves(bytes, "MatrixD mldivide", 8L * N * N, () -> sqd.mldivide(bd), () -> sqd.copy().mldivide(bd));

        MatrixF sqf = Matrices.view(dominantF(N + 1, SEED), 1, 1, N, N);
        MatrixF tllf = Matrices.view(Matrices.randomUniformF(N + 1, h + 1, SEED), 1, 1, N, h);
        MatrixF bf = Matrices.randomUniformF(N, 1, SEED + 1);
        assertSaves(bytes, "MatrixF solve square", 4L * N * N, () -> sqf.solve(bf, Matrices.createF(N, 1)),
                () -> sqf.copy().solve(bf, Matrices.createF(N, 1)));
        assertSaves(bytes, "MatrixF solve tall", 4L * N * h, () -> tllf.solve(bf, Matrices.createF(h, 1)),
                () -> tllf.copy().solve(bf, Matrices.createF(h, 1)));
        assertSaves(bytes, "MatrixF inv", 4L * N * N, () -> sqf.inv(Matrices.createF(N, N)),
                () -> sqf.copy().inv(Matrices.createF(N, N)));
        assertSaves(bytes, "MatrixF inverse", 4L * N * N, () -> sqf.inverse(), () -> sqf.copy().inverse());
        assertSaves(bytes, "MatrixF mldivide", 4L * N * N, () -> sqf.mldivide(bf), () -> sqf.copy().mldivide(bf));

        ComplexMatrixD sqzd = Matrices.view(dominantComplexD(N + 1, SEED), 1, 1, N, N);
        ComplexMatrixD tllzd = Matrices.view(Matrices.randomUniformComplexD(N + 1, h + 1, SEED), 1, 1, N, h);
        ComplexMatrixD bzd = Matrices.randomUniformComplexD(N, 1, SEED + 1);
        assertSaves(bytes, "ComplexMatrixD solve square", 16L * N * N,
                () -> sqzd.solve(bzd, Matrices.createComplexD(N, 1)),
                () -> sqzd.copy().solve(bzd, Matrices.createComplexD(N, 1)));
        assertSaves(bytes, "ComplexMatrixD solve tall", 16L * N * h,
                () -> tllzd.solve(bzd, Matrices.createComplexD(h, 1)),
                () -> tllzd.copy().solve(bzd, Matrices.createComplexD(h, 1)));
        assertSaves(bytes, "ComplexMatrixD inv", 16L * N * N, () -> sqzd.inv(Matrices.createComplexD(N, N)),
                () -> sqzd.copy().inv(Matrices.createComplexD(N, N)));
        assertSaves(bytes, "ComplexMatrixD inverse", 16L * N * N, () -> sqzd.inverse(), () -> sqzd.copy().inverse());
        assertSaves(bytes, "ComplexMatrixD mldivide", 16L * N * N,
                () -> sqzd.mldivide(bzd), () -> sqzd.copy().mldivide(bzd));

        ComplexMatrixF sqzf = Matrices.view(dominantComplexF(N + 1, SEED), 1, 1, N, N);
        ComplexMatrixF tllzf = Matrices.view(Matrices.randomUniformComplexF(N + 1, h + 1, SEED), 1, 1, N, h);
        ComplexMatrixF bzf = Matrices.randomUniformComplexF(N, 1, SEED + 1);
        assertSaves(bytes, "ComplexMatrixF solve square", 8L * N * N,
                () -> sqzf.solve(bzf, Matrices.createComplexF(N, 1)),
                () -> sqzf.copy().solve(bzf, Matrices.createComplexF(N, 1)));
        assertSaves(bytes, "ComplexMatrixF solve tall", 8L * N * h,
                () -> tllzf.solve(bzf, Matrices.createComplexF(h, 1)),
                () -> tllzf.copy().solve(bzf, Matrices.createComplexF(h, 1)));
        assertSaves(bytes, "ComplexMatrixF inv", 8L * N * N, () -> sqzf.inv(Matrices.createComplexF(N, N)),
                () -> sqzf.copy().inv(Matrices.createComplexF(N, N)));
        assertSaves(bytes, "ComplexMatrixF inverse", 8L * N * N, () -> sqzf.inverse(), () -> sqzf.copy().inverse());
        assertSaves(bytes, "ComplexMatrixF mldivide", 8L * N * N,
                () -> sqzf.mldivide(bzf), () -> sqzf.copy().mldivide(bzf));
    }

    // ---------------------------------------------------------------- timesMany

    @Test
    public void testTimesManyOnAViewCopiesOnlyForALongerChain() {
        MatrixDViewArgumentTest.CountingMatrixD pd = MatrixDViewArgumentTest.counting(10, 9, SEED + 9);
        MatrixD vd = Matrices.view(pd, 2, 1, 7, 5);
        MatrixD bd = Matrices.randomUniformD(5, 4, SEED + 10);
        MatrixD cd = Matrices.randomUniformD(4, 3, SEED + 11);
        MatrixD dd = Matrices.randomUniformD(3, 2, SEED + 12);
        MatrixD[] gotd = { vd.timesMany(bd), vd.timesMany(bd, cd) };
        assertEquals("MatrixD two and three factors: copies", 0, pd.copies);
        MatrixD fourd = vd.timesMany(bd, cd, dd);
        assertEquals("MatrixD four factors: copies", 1, pd.copies);
        assertClose("MatrixD two factors", vd.copy().times(bd), gotd[0]);
        assertClose("MatrixD three factors", vd.copy().times(bd).times(cd), gotd[1]);
        assertClose("MatrixD four factors", vd.copy().times(bd).times(cd).times(dd), fourd);

        MatrixFViewArgumentTest.CountingMatrixF pf = MatrixFViewArgumentTest.counting(10, 9, SEED + 9);
        MatrixF vf = Matrices.view(pf, 2, 1, 7, 5);
        MatrixF bf = Matrices.randomUniformF(5, 4, SEED + 10);
        MatrixF cf = Matrices.randomUniformF(4, 3, SEED + 11);
        MatrixF df = Matrices.randomUniformF(3, 2, SEED + 12);
        MatrixF[] gotf = { vf.timesMany(bf), vf.timesMany(bf, cf) };
        assertEquals("MatrixF two and three factors: copies", 0, pf.copies);
        MatrixF fourf = vf.timesMany(bf, cf, df);
        assertEquals("MatrixF four factors: copies", 1, pf.copies);
        assertClose("MatrixF two factors", vf.copy().times(bf), gotf[0]);
        assertClose("MatrixF three factors", vf.copy().times(bf).times(cf), gotf[1]);
        assertClose("MatrixF four factors", vf.copy().times(bf).times(cf).times(df), fourf);

        // a complex view is read in place only at the parent's origin
        ComplexMatrixDViewArgumentTest.CountingComplexMatrixD pzd = ComplexMatrixDViewArgumentTest.counting(10, 9,
                SEED + 9);
        ComplexMatrixD vzd = Matrices.view(pzd, 0, 0, 5, 4);
        ComplexMatrixD bzd = Matrices.randomUniformComplexD(5, 4, SEED + 10);
        ComplexMatrixD czd = Matrices.randomUniformComplexD(4, 3, SEED + 11);
        ComplexMatrixD dzd = Matrices.randomUniformComplexD(3, 2, SEED + 12);
        ComplexMatrixD[] gotzd = { vzd.timesMany(bzd), vzd.timesMany(bzd, czd) };
        assertEquals("ComplexMatrixD two and three factors: copies", 0, pzd.copies);
        ComplexMatrixD fourzd = vzd.timesMany(bzd, czd, dzd);
        assertEquals("ComplexMatrixD four factors: copies", 1, pzd.copies);
        assertClose("ComplexMatrixD two factors", vzd.copy().times(bzd), gotzd[0]);
        assertClose("ComplexMatrixD three factors", vzd.copy().times(bzd).times(czd), gotzd[1]);
        assertClose("ComplexMatrixD four factors", vzd.copy().times(bzd).times(czd).times(dzd), fourzd);

        ComplexMatrixFViewArgumentTest.CountingComplexMatrixF pzf = ComplexMatrixFViewArgumentTest.counting(10, 9,
                SEED + 9);
        ComplexMatrixF vzf = Matrices.view(pzf, 0, 0, 5, 4);
        ComplexMatrixF bzf = Matrices.randomUniformComplexF(5, 4, SEED + 10);
        ComplexMatrixF czf = Matrices.randomUniformComplexF(4, 3, SEED + 11);
        ComplexMatrixF dzf = Matrices.randomUniformComplexF(3, 2, SEED + 12);
        ComplexMatrixF[] gotzf = { vzf.timesMany(bzf), vzf.timesMany(bzf, czf) };
        assertEquals("ComplexMatrixF two and three factors: copies", 0, pzf.copies);
        ComplexMatrixF fourzf = vzf.timesMany(bzf, czf, dzf);
        assertEquals("ComplexMatrixF four factors: copies", 1, pzf.copies);
        assertClose("ComplexMatrixF two factors", vzf.copy().times(bzf), gotzf[0]);
        assertClose("ComplexMatrixF three factors", vzf.copy().times(bzf).times(czf), gotzf[1]);
        assertClose("ComplexMatrixF four factors", vzf.copy().times(bzf).times(czf).times(dzf), fourzf);
    }

    // ---------------------------------------------------------------- helpers

    static MatrixD dominantD(int n, long seed) {
        return Matrices.randomUniformD(n, n, seed).addInplace(20.0, Matrices.identityD(n));
    }

    static MatrixF dominantF(int n, long seed) {
        return Matrices.randomUniformF(n, n, seed).addInplace(20.0f, Matrices.identityF(n));
    }

    static ComplexMatrixD dominantComplexD(int n, long seed) {
        return Matrices.randomUniformComplexD(n, n, seed).addInplace(20.0, 0.0, Matrices.identityComplexD(n));
    }

    static ComplexMatrixF dominantComplexF(int n, long seed) {
        return Matrices.randomUniformComplexF(n, n, seed).addInplace(20.0f, 0.0f, Matrices.identityComplexF(n));
    }

    // the allocation counter of HotSpot's ThreadMXBean, or null
    static Method allocatedBytes() {
        try {
            Class<?> type = Class.forName("com.sun.management.ThreadMXBean");
            ThreadMXBean mx = ManagementFactory.getThreadMXBean();
            if (!type.isInstance(mx)) {
                return null;
            }
            Method m = type.getMethod("getThreadAllocatedBytes", long.class);
            return ((Long) m.invoke(mx, Thread.currentThread().getId())) >= 0L ? m : null;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    static long allocated(Method bytes, Runnable call) {
        ThreadMXBean mx = ManagementFactory.getThreadMXBean();
        long id = Thread.currentThread().getId();
        long least = Long.MAX_VALUE;
        try {
            for (int i = 0; i < 3; ++i) {
                long before = (Long) bytes.invoke(mx, id);
                call.run();
                least = Math.min(least, (Long) bytes.invoke(mx, id) - before);
            }
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(e);
        }
        return least;
    }

    // the view call must allocate at least half an array of the view's size less
    static void assertSaves(Method bytes, String what, long size, Runnable view, Runnable viaCopy) {
        for (int i = 0; i < 2; ++i) {
            view.run();
            viaCopy.run();
        }
        long v = allocated(bytes, view);
        long c = allocated(bytes, viaCopy);
        assertTrue(what + ": view " + v + " vs copy " + c + " bytes", c - v >= size / 2);
    }

    static void assertClose(String what, MatrixD want, MatrixD got) {
        assertEquals(what + ": rows", want.numRows(), got.numRows());
        assertEquals(what + ": cols", want.numColumns(), got.numColumns());
        assertClose(what, want.getArrayUnsafe(), got.getArrayUnsafe());
    }

    static void assertClose(String what, MatrixF want, MatrixF got) {
        assertEquals(what + ": rows", want.numRows(), got.numRows());
        assertEquals(what + ": cols", want.numColumns(), got.numColumns());
        assertClose(what, want.getArrayUnsafe(), got.getArrayUnsafe());
    }

    static void assertClose(String what, ComplexMatrixD want, ComplexMatrixD got) {
        assertEquals(what + ": rows", want.numRows(), got.numRows());
        assertEquals(what + ": cols", want.numColumns(), got.numColumns());
        assertClose(what, want.getArrayUnsafe(), got.getArrayUnsafe());
    }

    static void assertClose(String what, ComplexMatrixF want, ComplexMatrixF got) {
        assertEquals(what + ": rows", want.numRows(), got.numRows());
        assertEquals(what + ": cols", want.numColumns(), got.numColumns());
        assertClose(what, want.getArrayUnsafe(), got.getArrayUnsafe());
    }

    // BLAS and LAPACK may round differently for a different memory alignment
    static void assertClose(String what, double[] want, double[] got) {
        assertEquals(what + ": length", want.length, got.length);
        for (int i = 0; i < want.length; ++i) {
            double scale = Math.max(1.0, Math.max(Math.abs(want[i]), Math.abs(got[i])));
            if (!(Math.abs(want[i] - got[i]) <= 1e-10 * scale)) {
                fail(what + ": index " + i + " expected " + want[i] + " but was " + got[i]);
            }
        }
    }

    static void assertClose(String what, float[] want, float[] got) {
        assertEquals(what + ": length", want.length, got.length);
        for (int i = 0; i < want.length; ++i) {
            float scale = Math.max(1.0f, Math.max(Math.abs(want[i]), Math.abs(got[i])));
            if (!(Math.abs(want[i] - got[i]) <= 1e-4f * scale)) {
                fail(what + ": index " + i + " expected " + want[i] + " but was " + got[i]);
            }
        }
    }

    static void assertBits(String what, double[] want, double[] got) {
        assertEquals(what + ": length", want.length, got.length);
        for (int i = 0; i < want.length; ++i) {
            if (Double.doubleToRawLongBits(want[i]) != Double.doubleToRawLongBits(got[i])) {
                fail(what + ": index " + i + " expected " + want[i] + " but was " + got[i]);
            }
        }
    }

    static void assertBits(String what, float[] want, float[] got) {
        assertEquals(what + ": length", want.length, got.length);
        for (int i = 0; i < want.length; ++i) {
            if (Float.floatToRawIntBits(want[i]) != Float.floatToRawIntBits(got[i])) {
                fail(what + ": index " + i + " expected " + want[i] + " but was " + got[i]);
            }
        }
    }
}
