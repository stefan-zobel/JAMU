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
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

import org.junit.Test;

/**
 * Tests for the read-only {@code MatrixD} view.
 */
public final class MatrixDViewTest {

    private static final long SEED = 20260914L;

    /** interface methods a view refuses */
    private static final Set<String> REFUSED = sigs("scaleInplace(double)", "addInplace(MatrixD)",
            "addInplace(double,MatrixD)", "zeroInplace()", "setInplace(MatrixD)", "setInplace(double,MatrixD)",
            "setColumnInplace(int,MatrixD)", "setInplaceUpperTrapezoidal(MatrixD)",
            "setInplaceLowerTrapezoidal(MatrixD)", "set(int,int,double)", "add(int,int,double)",
            "setSubmatrixInplace(int,int,MatrixD,int,int,int,int)", "clampInplace(double,double)",
            "mapInplace(DFunction)", "addBroadcastedVectorInplace(MatrixD)", "mulBroadcastedVectorInplace(MatrixD)",
            "divBroadcastedVectorInplace(MatrixD)", "addBroadcastedRowVectorInplace(MatrixD)",
            "mulBroadcastedRowVectorInplace(MatrixD)", "divBroadcastedRowVectorInplace(MatrixD)",
            "zeroizeSubEpsilonInplace(int)", "zeroizeSubEpsilonRelativeInplace(int)",
            "sanitizeNonFiniteInplace(double,double,double)", "sanitizeNaNInplace(double)", "setUnsafe(int,int,double)",
            "getArrayUnsafe()");

    /** interface methods a view answers on a copy */
    private static final Set<String> COPY_READS = sigs("scale(double,MatrixD)", "trans(MatrixD)",
            "add(MatrixD,MatrixD)", "add(double,MatrixD,MatrixD)", "solve(MatrixD,MatrixD)",
            "inv(MatrixD)", "pseudoInv()", "expm()", "hadamard(MatrixD,MatrixD)", "toJaggedArray()", "normF()",
            "normMaxAbs()", "normInf()", "norm1()", "trace()", "selectColumn(int)", "selectConsecutiveColumns(int,int)",
            "selectSubmatrix(int,int,int,int)", "appendColumn(MatrixD)", "appendMatrix(MatrixD)", "mldivide(MatrixD)",
            "mrdivide(MatrixD)", "timesMany(MatrixD,MatrixD[])", "times(ComplexMatrixD)", "plus(MatrixD)",
            "minus(MatrixD)", "uminus()", "abs()", "transpose()", "inverse()", "hadamard(MatrixD)",
            "hadamardTransposed(MatrixD)", "transposedHadamard(MatrixD)", "map(DFunction)",
            "plusBroadcastedVector(MatrixD)", "mulBroadcastedVector(MatrixD)", "divBroadcastedVector(MatrixD)",
            "plusBroadcastedRowVector(MatrixD)", "mulBroadcastedRowVector(MatrixD)",
            "divBroadcastedRowVector(MatrixD)", "reshape(int,int)", "toComplexMatrix()");

    /** products a view computes with gemm on its parent's array */
    private static final Set<String> GEMM_READS = sigs("mult(MatrixD,MatrixD)", "mult(double,MatrixD,MatrixD)",
            "multAdd(MatrixD,MatrixD)", "multAdd(double,MatrixD,MatrixD)", "transABmult(MatrixD,MatrixD)",
            "transABmult(double,MatrixD,MatrixD)", "transAmult(MatrixD,MatrixD)", "transAmult(double,MatrixD,MatrixD)",
            "transBmult(MatrixD,MatrixD)", "transBmult(double,MatrixD,MatrixD)", "transABmultAdd(MatrixD,MatrixD)",
            "transABmultAdd(double,MatrixD,MatrixD)", "transAmultAdd(MatrixD,MatrixD)",
            "transAmultAdd(double,MatrixD,MatrixD)", "transBmultAdd(MatrixD,MatrixD)",
            "transBmultAdd(double,MatrixD,MatrixD)", "times(MatrixD)", "timesTimes(MatrixD,MatrixD)",
            "timesTransposed()", "timesTransposed(MatrixD)", "transposedTimes()", "transposedTimes(MatrixD)",
            "timesPlus(MatrixD,MatrixD)", "timesMinus(MatrixD,MatrixD)");

    /** interface methods a view answers from its parent, or by handing itself to a copying decomposition */
    private static final Set<String> DIRECT_READS = sigs("get(int,int)", "getUnsafe(int,int)", "toScalar()", "copy()",
            "submatrix(int,int,int,int,MatrixD,int,int)", "svd(boolean)", "svdEcon()", "singularValues()",
            "evd(boolean)", "qrd()", "lud()", "norm2()");

    /** default methods a view inherits on purpose */
    private static final Set<String> INHERITED = sigs("view(int,int,int,int)", "viewRows(int,int)",
            "viewColumns(int,int)");

    /** interface methods inherited from DimensionsBase */
    private static final Set<String> DIMENSIONS = sigs("numColumns()", "numRows()", "isScalar()", "isColumnVector()",
            "isRowVector()", "isSquareMatrix()", "startRow()", "endRow()", "startCol()", "endCol()", "isComplex()",
            "isDoublePrecision()", "checkIndex(int,int)", "checkSubmatrixIndexes(int,int,int,int)", "asString()",
            "getFormatString()", "setFormatString(String)");

    /** parent shapes, including heights around the block copy threshold */
    private static final int[][] SHAPES = { { 1, 1 }, { 1, 5 }, { 5, 1 }, { 3, 4 }, { 39, 7 }, { 40, 6 },
            { 41, 5 }, { 64, 64 } };

    @Test
    public void testElementsMatchTheParentRegion() {
        Random rnd = new Random(SEED);
        for (int[] s : SHAPES) {
            MatrixD A = Matrices.randomUniformD(s[0], s[1], rnd.nextLong());
            for (int draw = 0; draw < 40; ++draw) {
                int[] reg = region(rnd, s[0], s[1]);
                MatrixD V = Matrices.view(A, reg[0], reg[1], reg[2], reg[3]);
                assertRegion(A, reg, V);
            }
        }
    }

    @Test
    public void testCopyMatchesTheParentRegionBitExact() {
        Random rnd = new Random(SEED + 1L);
        for (int[] s : SHAPES) {
            MatrixD A = Matrices.randomUniformD(s[0], s[1], rnd.nextLong());
            for (int draw = 0; draw < 40; ++draw) {
                int[] reg = region(rnd, s[0], s[1]);
                MatrixD C = Matrices.view(A, reg[0], reg[1], reg[2], reg[3]).copy();
                assertTrue(C instanceof SimpleMatrixD);
                assertRegion(A, reg, C);
            }
        }
        // every block height on both sides of the threshold
        MatrixD A = Matrices.randomUniformD(48, 5, SEED);
        for (int len = 1; len <= 47; ++len) {
            int[] reg = { 1, 1, len, 3 };
            assertRegion(A, reg, Matrices.view(A, 1, 1, len, 3).copy());
        }
    }

    @Test
    public void testCopyIsDetached() {
        MatrixD A = Matrices.randomUniformD(6, 5, SEED);
        MatrixD V = Matrices.view(A, 1, 1, 4, 3);
        MatrixD C = V.copy();
        assertNotSame(V, C);
        double before = A.get(2, 2);
        C.set(1, 1, 42.0);
        assertBits("parent after writing the copy", before, A.get(2, 2));
        A.set(2, 2, -7.0);
        assertBits("copy after writing the parent", 42.0, C.get(1, 1));
    }

    @Test
    public void testViewIsLive() {
        MatrixD A = Matrices.randomUniformD(6, 5, SEED);
        MatrixD V = Matrices.view(A, 2, 1, 5, 4);
        A.set(3, 2, -0.0);
        assertBits("live read", -0.0, V.get(1, 1));
        assertBits("live unsafe read", -0.0, V.getUnsafe(1, 1));
    }

    @Test
    public void testViewOfViewReadsTheOriginalRegion() {
        MatrixD A = Matrices.randomUniformD(9, 8, SEED);
        MatrixD V = Matrices.view(A, 1, 2, 7, 6);
        MatrixD W = Matrices.view(V, 2, 1, 4, 3);
        assertTrue(W instanceof MatrixDView);
        assertRegion(A, new int[] { 3, 3, 5, 5 }, W);
        A.set(4, 4, 99.0);
        assertBits("live through a view of a view", 99.0, W.get(1, 1));
    }

    @Test
    public void testIllegalRegionsAreRejected() {
        MatrixD A = Matrices.randomUniformD(4, 3, SEED);
        try {
            Matrices.view((MatrixD) null, 0, 0, 0, 0);
            fail("null parent");
        } catch (NullPointerException expected) {
        }
        int[][] bad = { { -1, 0, 1, 1 }, { 0, -1, 1, 1 }, { 0, 0, 4, 1 }, { 0, 0, 1, 3 }, { 2, 0, 1, 1 },
                { 0, 2, 1, 1 } };
        for (int[] b : bad) {
            try {
                Matrices.view(A, b[0], b[1], b[2], b[3]);
                fail("region " + b[0] + ", " + b[1] + ", " + b[2] + ", " + b[3]);
            } catch (IllegalArgumentException expected) {
            }
        }
    }

    @Test
    public void testViewMethodsShowTheRegion() {
        Random rnd = new Random(SEED);
        for (int[] shape : SHAPES) {
            int rows = shape[0];
            int cols = shape[1];
            MatrixD A = Matrices.randomUniformD(rows, cols, SEED + 100 * rows + cols);
            for (int draw = 0; draw < 20; ++draw) {
                int[] reg = region(rnd, rows, cols);
                MatrixD V = A.view(reg[0], reg[1], reg[2], reg[3]);
                MatrixD R = A.viewRows(reg[0], reg[2]);
                MatrixD C = A.viewColumns(reg[1], reg[3]);
                assertRegion(A, reg, V);
                assertRegion(A, new int[] { reg[0], 0, reg[2], cols - 1 }, R);
                assertRegion(A, new int[] { 0, reg[1], rows - 1, reg[3] }, C);
                assertBitsArray("viewRows", A.selectSubmatrix(reg[0], 0, reg[2], cols - 1).getArrayUnsafe(),
                        R.copy().getArrayUnsafe());
                assertBitsArray("viewColumns", A.selectConsecutiveColumns(reg[1], reg[3]).getArrayUnsafe(),
                        C.copy().getArrayUnsafe());
                assertTrue(V instanceof MatrixDView && R instanceof MatrixDView && C instanceof MatrixDView);
            }
            MatrixD V = A.view(0, 0, rows - 1, cols - 1);
            MatrixD R = A.viewRows(rows - 1, rows - 1);
            MatrixD C = A.viewColumns(cols - 1, cols - 1);
            A.set(rows - 1, cols - 1, 12345.0);
            assertBits("live view", 12345.0, V.get(rows - 1, cols - 1));
            assertBits("live viewRows", 12345.0, R.get(0, cols - 1));
            assertBits("live viewColumns", 12345.0, C.get(rows - 1, 0));
            for (MatrixD M : new MatrixD[] { V, R, C }) {
                try {
                    M.set(0, 0, 1.0);
                    fail("set on a view");
                } catch (UnsupportedOperationException expected) {
                }
            }
        }
    }

    @Test
    public void testViewMethodsOnAViewReadTheOriginalParent() {
        MatrixD P = Matrices.randomUniformD(10, 11, SEED);
        MatrixD V = Matrices.view(P, 2, 3, 7, 8);
        MatrixD R = V.viewRows(1, 2);
        MatrixD C = V.viewColumns(1, 4);
        MatrixD W = V.view(1, 1, 3, 2);
        assertTrue(R instanceof MatrixDView && C instanceof MatrixDView && W instanceof MatrixDView);
        assertRegion(P, new int[] { 3, 3, 4, 8 }, R);
        assertRegion(P, new int[] { 2, 4, 7, 7 }, C);
        assertRegion(P, new int[] { 3, 4, 5, 5 }, W);
        P.set(3, 4, 99.0);
        assertBits("live viewRows", 99.0, R.get(0, 1));
        assertBits("live viewColumns", 99.0, C.get(1, 0));
        assertBits("live view", 99.0, W.get(0, 0));
    }

    @Test
    public void testViewMethodsRejectIllegalRanges() {
        MatrixD A = Matrices.randomUniformD(4, 3, SEED);
        int[][] rowRanges = { { -1, 0 }, { 0, 4 }, { 2, 1 } };
        for (int[] r : rowRanges) {
            String what = "viewRows(" + r[0] + ", " + r[1] + ")";
            assertEquals(what, IllegalArgumentException.class, exception(() -> A.viewRows(r[0], r[1])));
            assertEquals(what, exception(() -> A.selectSubmatrix(r[0], 0, r[1], 2)),
                    exception(() -> A.viewRows(r[0], r[1])));
        }
        int[][] colRanges = { { -1, 0 }, { 0, 3 }, { 2, 1 } };
        for (int[] c : colRanges) {
            String what = "viewColumns(" + c[0] + ", " + c[1] + ")";
            assertEquals(what, IllegalArgumentException.class, exception(() -> A.viewColumns(c[0], c[1])));
            assertEquals(what, exception(() -> A.selectConsecutiveColumns(c[0], c[1])),
                    exception(() -> A.viewColumns(c[0], c[1])));
        }
        int[][] regions = { { 0, 0, 4, 1 }, { 0, 0, 1, 3 }, { 1, 1, 0, 0 } };
        for (int[] b : regions) {
            String what = "view(" + b[0] + ", " + b[1] + ", " + b[2] + ", " + b[3] + ")";
            assertEquals(what, IllegalArgumentException.class, exception(() -> A.view(b[0], b[1], b[2], b[3])));
            assertEquals(what, exception(() -> A.selectSubmatrix(b[0], b[1], b[2], b[3])),
                    exception(() -> A.view(b[0], b[1], b[2], b[3])));
        }
    }

    @Test
    public void testGetChecksTheViewBoundsNotTheParentBounds() {
        MatrixD A = Matrices.randomUniformD(6, 6, SEED);
        MatrixD V = Matrices.view(A, 1, 1, 3, 3);
        assertEquals(3, V.numRows());
        assertEquals(3, V.numColumns());
        for (int[] ix : new int[][] { { 3, 0 }, { 0, 3 }, { -1, 0 }, { 0, -1 } }) {
            try {
                V.get(ix[0], ix[1]);
                fail("index " + ix[0] + ", " + ix[1]);
            } catch (IllegalArgumentException expected) {
            }
        }
    }

    @Test
    public void testToScalar() {
        MatrixD A = Matrices.randomUniformD(3, 3, SEED);
        assertBits("scalar", A.get(2, 1), Matrices.view(A, 2, 1, 2, 1).toScalar());
        try {
            Matrices.view(A, 0, 0, 1, 0).toScalar();
            fail("not a scalar");
        } catch (IllegalStateException expected) {
        }
    }

    @Test
    public void testToStringMatchesTheCopy() {
        MatrixD A = Matrices.randomUniformD(5, 4, SEED);
        MatrixD V = Matrices.view(A, 1, 0, 3, 2);
        assertEquals(V.copy().toString(), V.toString());
    }

    @Test
    public void testEveryMutatorThrowsAndLeavesEverythingUntouched() {
        RefusalFixture f = new RefusalFixture();
        List<Refusal> refusals = f.refusals();
        assertEquals(26, refusals.size());
        Set<String> names = new HashSet<>();
        for (Refusal r : refusals) {
            assertTrue("duplicate " + r.name, names.add(r.name));
            try {
                r.call.accept(f.V);
                fail(r.name + " must throw");
            } catch (UnsupportedOperationException expected) {
            }
            f.assertUntouched(r.name);
        }
    }

    @Test
    public void testRefusalComesBeforeArgumentChecks() {
        MatrixD V = Matrices.view(Matrices.randomUniformD(6, 5, SEED), 1, 1, 4, 3);
        List<Refusal> bad = new ArrayList<>();
        bad.add(new Refusal("set out of range", m -> m.set(99, 99, 1.0)));
        bad.add(new Refusal("setUnsafe out of range", m -> m.setUnsafe(-1, -1, 1.0)));
        bad.add(new Refusal("mapInplace null", m -> m.mapInplace(null)));
        bad.add(new Refusal("addInplace null", m -> m.addInplace(null)));
        bad.add(new Refusal("setSubmatrixInplace null", m -> m.setSubmatrixInplace(0, 0, null, 0, 0, 0, 0)));
        for (Refusal r : bad) {
            try {
                r.call.accept(V);
                fail(r.name + " must throw");
            } catch (UnsupportedOperationException expected) {
            } catch (RuntimeException e) {
                fail(r.name + " threw " + e);
            }
        }
    }

    @Test
    public void testTheViewStillReadsAfterRefusals() {
        RefusalFixture f = new RefusalFixture();
        for (Refusal r : f.refusals()) {
            try {
                r.call.accept(f.V);
            } catch (UnsupportedOperationException expected) {
            }
        }
        assertRegion(f.A, new int[] { 1, 1, 4, 3 }, f.V);
    }

    @Test
    public void testEveryCopyReadAgreesWithTheMaterializedRegion() {
        ReadFixture f = new ReadFixture();
        List<Read> reads = f.reads();
        assertEquals(66, reads.size());
        Set<String> names = new HashSet<>();
        for (Read r : reads) {
            assertTrue("duplicate " + r.name, names.add(r.name));
            MatrixD view = r.square ? f.Q : f.V;
            MatrixD expected = r.square ? f.EQ : f.E;
            Object got = r.call.apply(view);
            Object want = r.call.apply(expected);
            assertAgree(r.name, r.exact, want, got);
            if (got instanceof MatrixD) {
                assertTrue(r.name + " returned a view", !(got instanceof MatrixDView));
                ((MatrixD) got).set(0, 0, 12345.0);
            }
            f.assertParentsUntouched(r.name);
        }
        f.assertArgumentsUntouched();
    }

    @Test
    public void testCopyReadsFollowTheLiveParent() {
        MatrixD G = Matrices.randomUniformD(9, 8, SEED);
        MatrixD V = Matrices.view(G, 1, 2, 6, 6);
        MatrixD B = Matrices.randomUniformD(6, 5, SEED + 1L);
        double before = V.normF();
        G.set(3, 4, 100.0);
        MatrixD E = G.selectSubmatrix(1, 2, 6, 6);
        assertBits("normF after a parent write", E.normF(), V.normF());
        assertTrue("normF must change", Double.doubleToRawLongBits(before) != Double
                .doubleToRawLongBits(V.normF()));
        assertBitsArray("plus after a parent write", E.plus(B).getArrayUnsafe(), V.plus(B).getArrayUnsafe());
    }

    @Test
    public void testCopyReadsReportShapeErrorsLikeTheCopy() {
        MatrixD G = Matrices.randomUniformD(9, 8, SEED);
        MatrixD V = Matrices.view(G, 1, 2, 6, 6);
        MatrixD E = G.selectSubmatrix(1, 2, 6, 6);
        MatrixD wrong = Matrices.randomUniformD(6, 4, SEED);
        List<Read> bad = new ArrayList<>();
        bad.add(new Read("plus wrong shape", true, false, m -> m.plus(wrong)));
        bad.add(new Read("inverse not square", true, false, m -> m.inverse()));
        bad.add(new Read("trace not square", true, false, m -> m.trace()));
        for (Read r : bad) {
            Class<?> want = thrown(r, E);
            assertTrue(r.name + " must throw on the copy", want != null);
            assertEquals(r.name, want, thrown(r, V));
        }
    }

    @Test
    public void testSubmatrixAgreesWithTheCopyBitExact() {
        Random rnd = new Random(SEED + 2L);
        for (int[] s : SHAPES) {
            MatrixD A = Matrices.randomUniformD(s[0], s[1], rnd.nextLong());
            for (int draw = 0; draw < 40; ++draw) {
                int[] reg = region(rnd, s[0], s[1]);
                MatrixD V = Matrices.view(A, reg[0], reg[1], reg[2], reg[3]);
                int[] sub = region(rnd, V.numRows(), V.numColumns());
                int h = sub[2] - sub[0] + 1;
                int w = sub[3] - sub[1] + 1;
                int extraR = rnd.nextInt(4);
                int extraC = rnd.nextInt(4);
                checkSubmatrix("draw " + draw, V, sub, Matrices.randomUniformD(h + extraR, w + extraC,
                        rnd.nextLong()), rnd.nextInt(extraR + 1), rnd.nextInt(extraC + 1));
            }
        }
        // every block height on both sides of the threshold
        MatrixD A = Matrices.randomUniformD(48, 5, SEED);
        MatrixD V = Matrices.view(A, 0, 1, 47, 4);
        for (int len = 1; len <= 47; ++len) {
            checkSubmatrix("len " + len, V, new int[] { 1, 0, len, 3 }, Matrices.randomUniformD(48, 4, SEED + len),
                    48 - len, 0);
        }
    }

    @Test
    public void testSubmatrixChecksTheViewBounds() {
        MatrixD A = Matrices.randomUniformD(8, 8, SEED);
        MatrixD V = Matrices.view(A, 2, 2, 5, 5);
        MatrixD C = V.copy();
        List<Read> bad = new ArrayList<>();
        bad.add(new Read("region outside the view", true, false,
                m -> m.submatrix(0, 0, 4, 4, Matrices.createD(8, 8), 0, 0)));
        bad.add(new Read("target too small", true, false,
                m -> m.submatrix(0, 0, 3, 3, Matrices.createD(2, 2), 0, 0)));
        bad.add(new Read("target position out of range", true, false,
                m -> m.submatrix(0, 0, 1, 1, Matrices.createD(4, 4), 3, 0)));
        for (Read r : bad) {
            Class<?> want = thrown(r, C);
            assertTrue(r.name + " must throw on the copy", want != null);
            assertEquals(r.name, want, thrown(r, V));
        }
    }

    @Test
    public void testDecompositionsAgreeWithTheCopy() {
        MatrixD T = Matrices.randomUniformD(10, 9, SEED).map(x -> x + 1.0);
        MatrixD S = Matrices.randomUniformD(9, 9, SEED + 1L).map(x -> x + 1.0).addInplace(20.0,
                Matrices.identityD(9));
        double[] t0 = T.getArrayUnsafe().clone();
        double[] s0 = S.getArrayUnsafe().clone();
        MatrixD V = Matrices.view(T, 2, 3, 8, 7);
        MatrixD C = V.copy();
        MatrixD Q = Matrices.view(S, 2, 2, 7, 7);
        MatrixD CQ = Q.copy();

        for (boolean full : new boolean[] { true, false }) {
            String what = "svd(" + full + ")";
            SvdD want = C.svd(full);
            SvdD got = V.svd(full);
            assertValues(what + " S", false, want.getS(), got.getS());
            assertEquals(what, want.hasSingularVectors(), got.hasSingularVectors());
            if (want.hasSingularVectors()) {
                assertClose(what + " U", want.getU(), got.getU());
                assertClose(what + " Vt", want.getVt(), got.getVt());
            }
        }
        SvdEconD econWant = C.svdEcon();
        SvdEconD econGot = V.svdEcon();
        assertValues("svdEcon S", false, econWant.getS(), econGot.getS());
        assertClose("svdEcon U", econWant.getU(), econGot.getU());
        assertClose("svdEcon Vt", econWant.getVt(), econGot.getVt());

        QrdD qrWant = C.qrd();
        QrdD qrGot = V.qrd();
        assertClose("qrd Q", qrWant.getQ(), qrGot.getQ());
        assertClose("qrd R", qrWant.getR(), qrGot.getR());

        assertLud("lud tall", C.lud(), V.lud());
        assertLud("lud square", CQ.lud(), Q.lud());

        assertValues("singularValues", false, C.singularValues(), V.singularValues());
        assertValue("norm2", false, C.norm2(), V.norm2());

        for (boolean full : new boolean[] { true, false }) {
            String what = "evd(" + full + ")";
            EvdD want = CQ.evd(full);
            EvdD got = Q.evd(full);
            assertEquals(what + " count", want.getEigenvalues().length, got.getEigenvalues().length);
            for (int i = 0; i < want.getEigenvalues().length; ++i) {
                assertValue(what + " re " + i, false, want.getEigenvalues()[i].re(), got.getEigenvalues()[i].re());
                assertValue(what + " im " + i, false, want.getEigenvalues()[i].im(), got.getEigenvalues()[i].im());
            }
            assertEquals(what, want.hasEigenvectors(), got.hasEigenvectors());
            if (want.hasEigenvectors()) {
                assertClose(what + " vectors", want.getEigenvectors(), got.getEigenvectors());
            }
        }

        assertBitsArray("parent T", t0, T.getArrayUnsafe());
        assertBitsArray("parent S", s0, S.getArrayUnsafe());
    }

    @Test
    public void testDecompositionsAreDetachedFromTheParent() {
        MatrixD T = Matrices.randomUniformD(10, 9, SEED).map(x -> x + 1.0);
        MatrixD V = Matrices.view(T, 2, 3, 8, 7);
        SvdEconD svd = V.svdEcon();
        QrdD qr = V.qrd();
        double[] u = svd.getU().getArrayUnsafe().clone();
        double[] sv = svd.getS().clone();
        double[] vt = svd.getVt().getArrayUnsafe().clone();
        double[] q = qr.getQ().getArrayUnsafe().clone();
        double[] r = qr.getR().getArrayUnsafe().clone();
        T.set(4, 5, 1000.0);
        assertBitsArray("svdEcon U", u, svd.getU().getArrayUnsafe());
        assertBitsArray("svdEcon S", sv, svd.getS());
        assertBitsArray("svdEcon Vt", vt, svd.getVt().getArrayUnsafe());
        assertBitsArray("qrd Q", q, qr.getQ().getArrayUnsafe());
        assertBitsArray("qrd R", r, qr.getR().getArrayUnsafe());
    }

    @Test
    public void testEvdRejectsANonSquareView() {
        MatrixD T = Matrices.randomUniformD(10, 9, SEED);
        MatrixD V = Matrices.view(T, 2, 3, 8, 7);
        Read evd = new Read("evd on a non-square view", false, false, m -> m.evd(true));
        assertEquals(IllegalArgumentException.class, thrown(evd, V.copy()));
        assertEquals(IllegalArgumentException.class, thrown(evd, V));
    }

    @Test
    public void testEveryInterfaceMethodIsClassified() throws Exception {
        assertEquals(26, REFUSED.size());
        assertEquals(42, COPY_READS.size());
        assertEquals(24, GEMM_READS.size());
        assertEquals(12, DIRECT_READS.size());
        assertEquals(17, DIMENSIONS.size());
        assertEquals(3, INHERITED.size());
        assertEquals(new RefusalFixture().refusals().size(), REFUSED.size());
        assertEquals(new ReadFixture().reads().size(), COPY_READS.size() + GEMM_READS.size());

        Set<String> all = new TreeSet<>();
        for (Set<String> s : Arrays.asList(REFUSED, COPY_READS, GEMM_READS, DIRECT_READS, DIMENSIONS, INHERITED)) {
            for (String sig : s) {
                assertTrue("classified twice: " + sig, all.add(sig));
            }
        }
        Set<String> declared = new TreeSet<>();
        for (Method m : MatrixD.class.getMethods()) {
            String sig = sig(m);
            declared.add(sig);
            assertEquals("default method " + sig, INHERITED.contains(sig), m.isDefault());
            assertTrue("not classified: " + sig, all.contains(sig));
            Class<?> owner = MatrixDView.class.getMethod(m.getName(), m.getParameterTypes()).getDeclaringClass();
            Class<?> want = DIMENSIONS.contains(sig) ? DimensionsBase.class
                    : INHERITED.contains(sig) ? MatrixDConduct.class : MatrixDView.class;
            assertEquals("declaring class of " + sig, want, owner);
            if (INHERITED.contains(sig)) {
                Class<?> simple = SimpleMatrixD.class.getMethod(m.getName(), m.getParameterTypes())
                        .getDeclaringClass();
                assertEquals("declaring class in SimpleMatrixD of " + sig, MatrixDConduct.class, simple);
            }
        }
        for (String sig : all) {
            assertTrue("not in the interface: " + sig, declared.contains(sig));
        }
        assertEquals(124, declared.size());

        MatrixD V = Matrices.view(Matrices.randomUniformD(6, 5, SEED), 1, 1, 4, 3);
        for (Method m : MatrixD.class.getMethods()) {
            if (!REFUSED.contains(sig(m))) {
                continue;
            }
            try {
                m.invoke(V, dummyArgs(m));
                fail(sig(m) + " must throw");
            } catch (InvocationTargetException e) {
                assertTrue(sig(m) + " threw " + e.getCause(), e.getCause() instanceof UnsupportedOperationException);
            }
        }
    }

    @Test
    public void testAViewAsArgumentGivesTheCopyResult() {
        MatrixD P = Matrices.randomUniformD(50, 9, SEED).map(x -> x + 1.0);
        double[] p0 = P.getArrayUnsafe().clone();
        MatrixD block = Matrices.view(P, 2, 1, 46, 5);
        MatrixD column = Matrices.view(P, 2, 6, 46, 6);
        MatrixD row = Matrices.view(P, 3, 0, 3, 4);
        Supplier<MatrixD> t455 = () -> Matrices.randomUniformD(45, 5, SEED + 1L);
        Supplier<MatrixD> t745 = () -> Matrices.randomUniformD(7, 45, SEED + 2L);
        Supplier<MatrixD> square = () -> Matrices.randomUniformD(45, 45, SEED + 3L).addInplace(50.0,
                Matrices.identityD(45));

        List<ArgCase> cases = new ArrayList<>();
        cases.add(new ArgCase("addInplace", true, block, t455, (t, b) -> t.addInplace(b)));
        cases.add(new ArgCase("plus", true, block, t455, (t, b) -> t.plus(b)));
        cases.add(new ArgCase("times", false, block, t745, (t, b) -> t.times(b)));
        cases.add(new ArgCase("setInplace", true, block, t455, (t, b) -> t.setInplace(b)));
        cases.add(new ArgCase("setSubmatrixInplace height 10", true, block, t455,
                (t, b) -> t.setSubmatrixInplace(0, 0, b, 0, 0, 9, 4)));
        cases.add(new ArgCase("setSubmatrixInplace height 45", true, block, t455,
                (t, b) -> t.setSubmatrixInplace(0, 0, b, 0, 0, 44, 4)));
        cases.add(new ArgCase("hadamard", true, block, t455, (t, b) -> t.hadamard(b)));
        cases.add(new ArgCase("appendColumn", true, column, t455, (t, b) -> t.appendColumn(b)));
        cases.add(new ArgCase("plusBroadcastedRowVector", true, row, t455, (t, b) -> t.plusBroadcastedRowVector(b)));
        cases.add(new ArgCase("Matrices.distance", true, block, t455, (t, b) -> Matrices.distance(t, b)));
        cases.add(new ArgCase("Matrices.sumColumns", true, block, t455, (t, b) -> Matrices.sumColumns(b)));
        cases.add(new ArgCase("solve", false, block, square, (t, b) -> t.solve(b, Matrices.createD(45, 5))));

        for (ArgCase c : cases) {
            MatrixD t1 = c.target.get();
            Object got = c.call.apply(t1, c.arg);
            Object want = c.call.apply(c.target.get(), c.arg.copy());
            assertAgree(c.name, c.exact, want, got);
            assertBitsArray(c.name + ": parent", p0, P.getArrayUnsafe());
        }
    }

    // ---------------------------------------------------------------- classification and arguments

    static final class ArgCase {
        final String name;
        final boolean exact;
        final MatrixD arg;
        final Supplier<MatrixD> target;
        final BiFunction<MatrixD, MatrixD, Object> call;

        ArgCase(String name, boolean exact, MatrixD arg, Supplier<MatrixD> target,
                BiFunction<MatrixD, MatrixD, Object> call) {
            this.name = name;
            this.exact = exact;
            this.arg = arg;
            this.target = target;
            this.call = call;
        }
    }

    static Set<String> sigs(String... sigs) {
        return new HashSet<>(Arrays.asList(sigs));
    }

    static String sig(Method m) {
        StringBuilder b = new StringBuilder(m.getName()).append('(');
        Class<?>[] p = m.getParameterTypes();
        for (int i = 0; i < p.length; ++i) {
            b.append(i == 0 ? "" : ",").append(p[i].getSimpleName());
        }
        return b.append(')').toString();
    }

    static Object[] dummyArgs(Method m) {
        Class<?>[] p = m.getParameterTypes();
        Object[] args = new Object[p.length];
        for (int i = 0; i < p.length; ++i) {
            if (p[i] == int.class) {
                args[i] = 0;
            } else if (p[i] == double.class) {
                args[i] = 0.0;
            } else if (p[i] == boolean.class) {
                args[i] = false;
            }
        }
        return args;
    }

    // ---------------------------------------------------------------- refusals

    static final class Refusal {
        final String name;
        final Consumer<MatrixD> call;

        Refusal(String name, Consumer<MatrixD> call) {
            this.name = name;
            this.call = call;
        }
    }

    // a 4 x 3 view on a 6 x 5 parent plus ordinary arguments of matching shape
    static final class RefusalFixture {
        final MatrixD A = Matrices.randomUniformD(6, 5, SEED);
        final MatrixD V = Matrices.view(A, 1, 1, 4, 3);
        final MatrixD B = Matrices.randomUniformD(4, 3, SEED + 1L);
        final MatrixD col = Matrices.randomUniformD(4, 1, SEED + 2L);
        final MatrixD row = Matrices.randomUniformD(1, 3, SEED + 3L);
        final double[] a0 = A.getArrayUnsafe().clone();
        final double[] b0 = B.getArrayUnsafe().clone();
        final double[] col0 = col.getArrayUnsafe().clone();
        final double[] row0 = row.getArrayUnsafe().clone();

        List<Refusal> refusals() {
            List<Refusal> l = new ArrayList<>();
            l.add(new Refusal("scaleInplace", m -> m.scaleInplace(2.0)));
            l.add(new Refusal("addInplace(B)", m -> m.addInplace(B)));
            l.add(new Refusal("addInplace(alpha, B)", m -> m.addInplace(2.0, B)));
            l.add(new Refusal("zeroInplace", m -> m.zeroInplace()));
            l.add(new Refusal("setInplace(other)", m -> m.setInplace(B)));
            l.add(new Refusal("setInplace(alpha, B)", m -> m.setInplace(2.0, B)));
            l.add(new Refusal("setColumnInplace", m -> m.setColumnInplace(0, col)));
            l.add(new Refusal("setInplaceUpperTrapezoidal", m -> m.setInplaceUpperTrapezoidal(B)));
            l.add(new Refusal("setInplaceLowerTrapezoidal", m -> m.setInplaceLowerTrapezoidal(B)));
            l.add(new Refusal("set", m -> m.set(0, 0, 1.0)));
            l.add(new Refusal("add(row, col, val)", m -> m.add(0, 0, 1.0)));
            l.add(new Refusal("setSubmatrixInplace", m -> m.setSubmatrixInplace(0, 0, B, 0, 0, 1, 1)));
            l.add(new Refusal("clampInplace", m -> m.clampInplace(0.25, 0.75)));
            l.add(new Refusal("mapInplace", m -> m.mapInplace(x -> x + 1.0)));
            l.add(new Refusal("addBroadcastedVectorInplace", m -> m.addBroadcastedVectorInplace(col)));
            l.add(new Refusal("mulBroadcastedVectorInplace", m -> m.mulBroadcastedVectorInplace(col)));
            l.add(new Refusal("divBroadcastedVectorInplace", m -> m.divBroadcastedVectorInplace(col)));
            l.add(new Refusal("addBroadcastedRowVectorInplace", m -> m.addBroadcastedRowVectorInplace(row)));
            l.add(new Refusal("mulBroadcastedRowVectorInplace", m -> m.mulBroadcastedRowVectorInplace(row)));
            l.add(new Refusal("divBroadcastedRowVectorInplace", m -> m.divBroadcastedRowVectorInplace(row)));
            l.add(new Refusal("zeroizeSubEpsilonInplace", m -> m.zeroizeSubEpsilonInplace(1)));
            l.add(new Refusal("zeroizeSubEpsilonRelativeInplace", m -> m.zeroizeSubEpsilonRelativeInplace(1)));
            l.add(new Refusal("sanitizeNonFiniteInplace", m -> m.sanitizeNonFiniteInplace(0.0, 1.0, -1.0)));
            l.add(new Refusal("sanitizeNaNInplace", m -> m.sanitizeNaNInplace(0.0)));
            l.add(new Refusal("setUnsafe", m -> m.setUnsafe(0, 0, 1.0)));
            l.add(new Refusal("getArrayUnsafe", m -> m.getArrayUnsafe()));
            return l;
        }

        void assertUntouched(String after) {
            assertBitsArray(after + ": parent", a0, A.getArrayUnsafe());
            assertBitsArray(after + ": B", b0, B.getArrayUnsafe());
            assertBitsArray(after + ": column vector", col0, col.getArrayUnsafe());
            assertBitsArray(after + ": row vector", row0, row.getArrayUnsafe());
        }
    }

    // ---------------------------------------------------------------- reads

    static final class Read {
        final String name;
        final boolean exact;
        final boolean square;
        final Function<MatrixD, Object> call;

        Read(String name, boolean exact, boolean square, Function<MatrixD, Object> call) {
            this.name = name;
            this.exact = exact;
            this.square = square;
            this.call = call;
        }
    }

    // a 6 x 5 view on a 9 x 8 parent, a 6 x 6 well conditioned view on a 9 x 9
    // parent, their materialized regions and shared ordinary arguments
    static final class ReadFixture {
        final MatrixD G = Matrices.randomUniformD(9, 8, SEED).map(x -> x + 1.0);
        final MatrixD S = Matrices.randomUniformD(9, 9, SEED + 1L).map(x -> x + 1.0)
                .addInplace(20.0, Matrices.identityD(9));
        final MatrixD V = Matrices.view(G, 1, 2, 6, 6);
        final MatrixD E = G.selectSubmatrix(1, 2, 6, 6);
        final MatrixD Q = Matrices.view(S, 2, 2, 7, 7);
        final MatrixD EQ = S.selectSubmatrix(2, 2, 7, 7);
        final double[] g0 = G.getArrayUnsafe().clone();
        final double[] s0 = S.getArrayUnsafe().clone();

        final List<MatrixD> args = new ArrayList<>();
        final List<double[]> args0 = new ArrayList<>();
        final MatrixD b65 = arg(6, 5);
        final MatrixD b54 = arg(5, 4);
        final MatrixD b64 = arg(6, 4);
        final MatrixD b45 = arg(4, 5);
        final MatrixD b46 = arg(4, 6);
        final MatrixD b56 = arg(5, 6);
        final MatrixD b43 = arg(4, 3);
        final MatrixD b32 = arg(3, 2);
        final MatrixD b62 = arg(6, 2);
        final MatrixD col6 = arg(6, 1);
        final MatrixD row5 = arg(1, 5);
        final MatrixD d66 = arg(6, 6).addInplace(20.0, Matrices.identityD(6));
        final MatrixD c64 = arg(6, 4);
        final MatrixD c54 = arg(5, 4);
        final ComplexMatrixD z53 = Matrices.randomUniformComplexD(5, 3, SEED + 99L);
        final double[] z530 = z53.getArrayUnsafe().clone();

        MatrixD arg(int rows, int cols) {
            MatrixD m = Matrices.randomUniformD(rows, cols, SEED + 10L + args.size()).map(x -> x + 1.0);
            args.add(m);
            return m;
        }

        // runs after the field initializers, so d66 is already shifted
        ReadFixture() {
            for (MatrixD m : args) {
                args0.add(m.getArrayUnsafe().clone());
            }
        }

        List<Read> reads() {
            List<Read> l = new ArrayList<>();
            l.add(new Read("scale(alpha, B)", true, false, m -> m.scale(2.5, out(6, 5))));
            l.add(new Read("trans", true, false, m -> m.trans(out(5, 6))));
            l.add(new Read("add(B, C)", true, false, m -> m.add(b65, out(6, 5))));
            l.add(new Read("add(alpha, B, C)", true, false, m -> m.add(2.5, b65, out(6, 5))));
            l.add(new Read("mult(B, C)", false, false, m -> m.mult(b54, out(6, 4))));
            l.add(new Read("mult(alpha, B, C)", false, false, m -> m.mult(2.5, b54, out(6, 4))));
            l.add(new Read("multAdd(B, C)", false, false, m -> m.multAdd(b54, c64.copy())));
            l.add(new Read("multAdd(alpha, B, C)", false, false, m -> m.multAdd(2.5, b54, c64.copy())));
            l.add(new Read("transABmult(B, C)", false, false, m -> m.transABmult(b46, out(5, 4))));
            l.add(new Read("transABmult(alpha, B, C)", false, false, m -> m.transABmult(2.5, b46, out(5, 4))));
            l.add(new Read("transAmult(B, C)", false, false, m -> m.transAmult(b64, out(5, 4))));
            l.add(new Read("transAmult(alpha, B, C)", false, false, m -> m.transAmult(2.5, b64, out(5, 4))));
            l.add(new Read("transBmult(B, C)", false, false, m -> m.transBmult(b45, out(6, 4))));
            l.add(new Read("transBmult(alpha, B, C)", false, false, m -> m.transBmult(2.5, b45, out(6, 4))));
            l.add(new Read("transABmultAdd(B, C)", false, false, m -> m.transABmultAdd(b46, c54.copy())));
            l.add(new Read("transABmultAdd(alpha, B, C)", false, false,
                    m -> m.transABmultAdd(2.5, b46, c54.copy())));
            l.add(new Read("transAmultAdd(B, C)", false, false, m -> m.transAmultAdd(b64, c54.copy())));
            l.add(new Read("transAmultAdd(alpha, B, C)", false, false,
                    m -> m.transAmultAdd(2.5, b64, c54.copy())));
            l.add(new Read("transBmultAdd(B, C)", false, false, m -> m.transBmultAdd(b45, c64.copy())));
            l.add(new Read("transBmultAdd(alpha, B, C)", false, false,
                    m -> m.transBmultAdd(2.5, b45, c64.copy())));
            l.add(new Read("solve", false, true, m -> m.solve(b62, out(6, 2))));
            l.add(new Read("inv", false, true, m -> m.inv(out(6, 6))));
            l.add(new Read("pseudoInv", false, false, m -> m.pseudoInv()));
            l.add(new Read("expm", false, true, m -> m.expm()));
            l.add(new Read("hadamard(B, out)", true, false, m -> m.hadamard(b65, out(6, 5))));
            l.add(new Read("toJaggedArray", true, false, m -> m.toJaggedArray()));
            l.add(new Read("normF", true, false, m -> m.normF()));
            l.add(new Read("normMaxAbs", true, false, m -> m.normMaxAbs()));
            l.add(new Read("normInf", true, false, m -> m.normInf()));
            l.add(new Read("norm1", true, false, m -> m.norm1()));
            l.add(new Read("trace", true, true, m -> m.trace()));
            l.add(new Read("selectColumn", true, false, m -> m.selectColumn(2)));
            l.add(new Read("selectConsecutiveColumns", true, false, m -> m.selectConsecutiveColumns(1, 3)));
            l.add(new Read("selectSubmatrix", true, false, m -> m.selectSubmatrix(1, 1, 4, 3)));
            l.add(new Read("appendColumn", true, false, m -> m.appendColumn(col6)));
            l.add(new Read("appendMatrix", true, false, m -> m.appendMatrix(b62)));
            l.add(new Read("mldivide", false, true, m -> m.mldivide(b62)));
            l.add(new Read("mrdivide", false, true, m -> m.mrdivide(d66)));
            l.add(new Read("times(MatrixD)", false, false, m -> m.times(b54)));
            l.add(new Read("timesTimes", false, false, m -> m.timesTimes(b54, b43)));
            l.add(new Read("timesMany", false, false, m -> m.timesMany(b54, b43, b32)));
            l.add(new Read("timesTransposed()", false, false, m -> m.timesTransposed()));
            l.add(new Read("timesTransposed(B)", false, false, m -> m.timesTransposed(b45)));
            l.add(new Read("transposedTimes()", false, false, m -> m.transposedTimes()));
            l.add(new Read("transposedTimes(B)", false, false, m -> m.transposedTimes(b64)));
            l.add(new Read("times(ComplexMatrixD)", false, false, m -> m.times(z53)));
            l.add(new Read("plus", true, false, m -> m.plus(b65)));
            l.add(new Read("timesPlus", false, false, m -> m.timesPlus(b54, c64)));
            l.add(new Read("timesMinus", false, false, m -> m.timesMinus(b54, c64)));
            l.add(new Read("minus", true, false, m -> m.minus(b65)));
            l.add(new Read("uminus", true, false, m -> m.uminus()));
            l.add(new Read("abs", true, false, m -> m.abs()));
            l.add(new Read("transpose", true, false, m -> m.transpose()));
            l.add(new Read("inverse", false, true, m -> m.inverse()));
            l.add(new Read("hadamard(B)", true, false, m -> m.hadamard(b65)));
            l.add(new Read("hadamardTransposed", true, false, m -> m.hadamardTransposed(b56)));
            l.add(new Read("transposedHadamard", true, false, m -> m.transposedHadamard(b56)));
            l.add(new Read("map", true, false, m -> m.map(x -> x * x + 1.0)));
            l.add(new Read("plusBroadcastedVector", true, false, m -> m.plusBroadcastedVector(col6)));
            l.add(new Read("mulBroadcastedVector", true, false, m -> m.mulBroadcastedVector(col6)));
            l.add(new Read("divBroadcastedVector", true, false, m -> m.divBroadcastedVector(col6)));
            l.add(new Read("plusBroadcastedRowVector", true, false, m -> m.plusBroadcastedRowVector(row5)));
            l.add(new Read("mulBroadcastedRowVector", true, false, m -> m.mulBroadcastedRowVector(row5)));
            l.add(new Read("divBroadcastedRowVector", true, false, m -> m.divBroadcastedRowVector(row5)));
            l.add(new Read("reshape", true, false, m -> m.reshape(5, 6)));
            l.add(new Read("toComplexMatrix", true, false, m -> m.toComplexMatrix()));
            return l;
        }

        static MatrixD out(int rows, int cols) {
            return Matrices.createD(rows, cols);
        }

        void assertParentsUntouched(String after) {
            assertBitsArray(after + ": parent G", g0, G.getArrayUnsafe());
            assertBitsArray(after + ": parent S", s0, S.getArrayUnsafe());
        }

        void assertArgumentsUntouched() {
            for (int i = 0; i < args.size(); ++i) {
                assertBitsArray("argument " + i, args0.get(i), args.get(i).getArrayUnsafe());
            }
            assertBitsArray("complex argument", z530, z53.getArrayUnsafe());
        }
    }

    // the class of the runtime exception thrown, or null
    static Class<?> exception(Runnable call) {
        try {
            call.run();
            return null;
        } catch (RuntimeException e) {
            return e.getClass();
        }
    }

    static Class<?> thrown(Read r, MatrixD m) {
        try {
            r.call.apply(m);
            return null;
        } catch (RuntimeException e) {
            return e.getClass();
        }
    }

    static void assertAgree(String what, boolean exact, Object want, Object got) {
        if (want instanceof Double) {
            assertValue(what, exact, (Double) want, (Double) got);
        } else if (want instanceof double[]) {
            assertValues(what, exact, (double[]) want, (double[]) got);
        } else if (want instanceof double[][]) {
            double[][] w = (double[][]) want;
            double[][] g = (double[][]) got;
            assertEquals(what + ": rows", w.length, g.length);
            for (int i = 0; i < w.length; ++i) {
                assertValues(what + " row " + i, exact, w[i], g[i]);
            }
        } else if (want instanceof MatrixD) {
            MatrixD w = (MatrixD) want;
            MatrixD g = (MatrixD) got;
            assertEquals(what + ": rows", w.numRows(), g.numRows());
            assertEquals(what + ": cols", w.numColumns(), g.numColumns());
            assertValues(what, exact, w.getArrayUnsafe(), g.getArrayUnsafe());
        } else if (want instanceof ComplexMatrixD) {
            ComplexMatrixD w = (ComplexMatrixD) want;
            ComplexMatrixD g = (ComplexMatrixD) got;
            assertEquals(what + ": rows", w.numRows(), g.numRows());
            assertEquals(what + ": cols", w.numColumns(), g.numColumns());
            assertValues(what, exact, w.getArrayUnsafe(), g.getArrayUnsafe());
        } else {
            fail(what + ": unexpected result type " + want);
        }
    }

    static void assertValues(String what, boolean exact, double[] want, double[] got) {
        assertEquals(what + ": length", want.length, got.length);
        for (int i = 0; i < want.length; ++i) {
            assertValue(what + " [" + i + "]", exact, want[i], got[i]);
        }
    }

    // BLAS and LAPACK may round differently for a different memory alignment
    static void assertValue(String what, boolean exact, double want, double got) {
        if (exact) {
            assertBits(what, want, got);
        } else {
            double scale = Math.max(1.0, Math.max(Math.abs(want), Math.abs(got)));
            assertTrue(what + ": " + want + " vs " + got, Math.abs(want - got) <= 1e-10 * scale);
        }
    }

    // ---------------------------------------------------------------- submatrix and decompositions

    // the view and its copy must write the same bits into equal targets
    static void checkSubmatrix(String what, MatrixD V, int[] sub, MatrixD target, int rb, int cb) {
        MatrixD B1 = target.copy();
        MatrixD B2 = target.copy();
        MatrixD got = V.submatrix(sub[0], sub[1], sub[2], sub[3], B1, rb, cb);
        assertTrue(what + ": must return the target", got == B1);
        V.copy().submatrix(sub[0], sub[1], sub[2], sub[3], B2, rb, cb);
        assertBitsArray(what, B2.getArrayUnsafe(), B1.getArrayUnsafe());
    }

    static void assertClose(String what, MatrixD want, MatrixD got) {
        assertEquals(what + ": rows", want.numRows(), got.numRows());
        assertEquals(what + ": cols", want.numColumns(), got.numColumns());
        assertValues(what, false, want.getArrayUnsafe(), got.getArrayUnsafe());
    }

    static void assertLud(String what, LudD want, LudD got) {
        // P is null when no rows were interchanged
        assertEquals(what + " P present", want.getP() != null, got.getP() != null);
        if (want.getP() != null) {
            assertClose(what + " P", want.getP(), got.getP());
        }
        assertClose(what + " L", want.getL(), got.getL());
        assertClose(what + " U", want.getU(), got.getU());
        assertEquals(what + " singular", want.isSingular(), got.isSingular());
    }

    // ---------------------------------------------------------------- helpers

    static int[] region(Random rnd, int rows, int cols) {
        int r0 = rnd.nextInt(rows);
        int r1 = r0 + rnd.nextInt(rows - r0);
        int c0 = rnd.nextInt(cols);
        int c1 = c0 + rnd.nextInt(cols - c0);
        return new int[] { r0, c0, r1, c1 };
    }

    // expected values come straight from the parent's backing array
    static void assertRegion(MatrixD A, int[] reg, MatrixD M) {
        double[] a = A.getArrayUnsafe();
        int rows = reg[2] - reg[0] + 1;
        int cols = reg[3] - reg[1] + 1;
        assertEquals(rows, M.numRows());
        assertEquals(cols, M.numColumns());
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                double want = a[(reg[1] + col) * A.numRows() + reg[0] + row];
                assertBits("(" + row + ", " + col + ")", want, M.get(row, col));
                assertBits("unsafe (" + row + ", " + col + ")", want, M.getUnsafe(row, col));
            }
        }
    }

    static void assertBits(String what, double want, double got) {
        assertEquals(what, Double.doubleToRawLongBits(want), Double.doubleToRawLongBits(got));
    }

    static void assertBitsArray(String what, double[] want, double[] got) {
        assertEquals(what + ": length", want.length, got.length);
        for (int i = 0; i < want.length; ++i) {
            assertBits(what + " [" + i + "]", want[i], got[i]);
        }
    }
}
