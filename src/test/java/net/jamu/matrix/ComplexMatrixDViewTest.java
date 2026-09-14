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

import net.jamu.complex.Zd;
import net.jamu.complex.ZdImpl;

/**
 * Tests for the read-only {@code ComplexMatrixD} view.
 */
public final class ComplexMatrixDViewTest {

    private static final long SEED = 20260916L;

    /** interface methods a view refuses */
    private static final Set<String> REFUSED = sigs("scaleInplace(double,double)", "addInplace(ComplexMatrixD)",
            "addInplace(double,double,ComplexMatrixD)", "zeroInplace()", "setInplace(ComplexMatrixD)",
            "setInplace(double,double,ComplexMatrixD)", "setColumnInplace(int,ComplexMatrixD)",
            "setInplaceUpperTrapezoidal(ComplexMatrixD)", "setInplaceLowerTrapezoidal(ComplexMatrixD)",
            "set(int,int,double,double)", "add(int,int,double,double)",
            "setSubmatrixInplace(int,int,ComplexMatrixD,int,int,int,int)", "zeroizeSubEpsilonInplace(int)",
            "zeroizeSubEpsilonRelativeInplace(int)", "sanitizeNonFiniteInplace(double,double,double)",
            "sanitizeNaNInplace(double)", "setUnsafe(int,int,double,double)", "getArrayUnsafe()");

    /** interface methods a view answers on a copy */
    private static final Set<String> COPY_READS = sigs("scale(double,double,ComplexMatrixD)",
            "conjTrans(ComplexMatrixD)", "trans(ComplexMatrixD)", "add(ComplexMatrixD,ComplexMatrixD)",
            "add(double,double,ComplexMatrixD,ComplexMatrixD)", "solve(ComplexMatrixD,ComplexMatrixD)",
            "inv(ComplexMatrixD)", "pseudoInv()", "expm()", "hadamard(ComplexMatrixD,ComplexMatrixD)",
            "toJaggedArray()", "normF()", "normMaxAbs()", "normInf()", "norm1()", "trace()", "selectColumn(int)",
            "selectConsecutiveColumns(int,int)", "selectSubmatrix(int,int,int,int)", "appendColumn(ComplexMatrixD)",
            "appendMatrix(ComplexMatrixD)", "mldivide(ComplexMatrixD)", "mrdivide(ComplexMatrixD)",
            "timesMany(ComplexMatrixD,ComplexMatrixD[])", "plus(ComplexMatrixD)", "minus(ComplexMatrixD)",
            "uminus()", "abs()", "conjugateTranspose()", "transpose()", "inverse()", "hadamard(ComplexMatrixD)",
            "reshape(int,int)", "toRealMatrix()");

    /** products a view computes with zgemm on its parent's array */
    private static final Set<String> GEMM_READS = sigs("mult(ComplexMatrixD,ComplexMatrixD)",
            "mult(double,double,ComplexMatrixD,ComplexMatrixD)", "multAdd(ComplexMatrixD,ComplexMatrixD)",
            "multAdd(double,double,ComplexMatrixD,ComplexMatrixD)", "conjTransABmult(ComplexMatrixD,ComplexMatrixD)",
            "conjTransABmult(double,double,ComplexMatrixD,ComplexMatrixD)",
            "conjTransAmult(ComplexMatrixD,ComplexMatrixD)",
            "conjTransAmult(double,double,ComplexMatrixD,ComplexMatrixD)",
            "conjTransBmult(ComplexMatrixD,ComplexMatrixD)",
            "conjTransBmult(double,double,ComplexMatrixD,ComplexMatrixD)",
            "conjTransABmultAdd(ComplexMatrixD,ComplexMatrixD)",
            "conjTransABmultAdd(double,double,ComplexMatrixD,ComplexMatrixD)",
            "conjTransAmultAdd(ComplexMatrixD,ComplexMatrixD)",
            "conjTransAmultAdd(double,double,ComplexMatrixD,ComplexMatrixD)",
            "conjTransBmultAdd(ComplexMatrixD,ComplexMatrixD)",
            "conjTransBmultAdd(double,double,ComplexMatrixD,ComplexMatrixD)", "times(ComplexMatrixD)",
            "timesTimes(ComplexMatrixD,ComplexMatrixD)", "timesConjugateTransposed()",
            "timesConjugateTransposed(ComplexMatrixD)", "conjugateTransposedTimes()",
            "conjugateTransposedTimes(ComplexMatrixD)", "times(MatrixD)", "timesPlus(ComplexMatrixD,ComplexMatrixD)",
            "timesMinus(ComplexMatrixD,ComplexMatrixD)");

    /** interface methods a view answers from its parent, or by handing itself to a decomposition */
    private static final Set<String> DIRECT_READS = sigs("get(int,int,Zd)", "get(int,int)", "getUnsafe(int,int,Zd)",
            "getUnsafe(int,int)", "toScalar()", "copy()", "submatrix(int,int,int,int,ComplexMatrixD,int,int)",
            "svd(boolean)", "svdEcon()", "singularValues()", "evd(boolean)", "qrd()", "lud()", "norm2()");

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
            ComplexMatrixD A = Matrices.randomUniformComplexD(s[0], s[1], rnd.nextLong());
            for (int draw = 0; draw < 40; ++draw) {
                int[] reg = region(rnd, s[0], s[1]);
                ComplexMatrixD V = Matrices.view(A, reg[0], reg[1], reg[2], reg[3]);
                assertTrue(V.isComplex());
                assertRegion(A, reg, V);
            }
        }
    }

    @Test
    public void testCopyMatchesTheParentRegionBitExact() {
        Random rnd = new Random(SEED + 1L);
        for (int[] s : SHAPES) {
            ComplexMatrixD A = Matrices.randomUniformComplexD(s[0], s[1], rnd.nextLong());
            for (int draw = 0; draw < 40; ++draw) {
                int[] reg = region(rnd, s[0], s[1]);
                ComplexMatrixD C = Matrices.view(A, reg[0], reg[1], reg[2], reg[3]).copy();
                assertTrue(C instanceof SimpleComplexMatrixD);
                assertRegion(A, reg, C);
            }
        }
        // every block height on both sides of the threshold
        ComplexMatrixD A = Matrices.randomUniformComplexD(48, 5, SEED);
        for (int len = 1; len <= 47; ++len) {
            int[] reg = { 1, 1, len, 3 };
            assertRegion(A, reg, Matrices.view(A, 1, 1, len, 3).copy());
        }
    }

    @Test
    public void testCopyIsDetached() {
        ComplexMatrixD A = Matrices.randomUniformComplexD(6, 5, SEED);
        ComplexMatrixD V = Matrices.view(A, 1, 1, 4, 3);
        ComplexMatrixD C = V.copy();
        assertNotSame(V, C);
        Zd before = A.get(2, 2);
        C.set(1, 1, 42.0, -42.0);
        assertBitsZ("parent after writing the copy", before, A.get(2, 2));
        A.set(2, 2, -7.0, 7.0);
        assertBitsZ("copy after writing the parent", new ZdImpl(42.0, -42.0), C.get(1, 1));
    }

    @Test
    public void testViewIsLive() {
        ComplexMatrixD A = Matrices.randomUniformComplexD(6, 5, SEED);
        ComplexMatrixD V = Matrices.view(A, 2, 1, 5, 4);
        A.set(3, 2, -0.0, 5.0);
        Zd want = new ZdImpl(-0.0, 5.0);
        assertBitsZ("live read", want, V.get(1, 1));
        assertBitsZ("live unsafe read", want, V.getUnsafe(1, 1));
        Zd out = new ZdImpl(0.0);
        V.get(1, 1, out);
        assertBitsZ("live read into", want, out);
        V.getUnsafe(1, 1, out);
        assertBitsZ("live unsafe read into", want, out);
    }

    @Test
    public void testViewOfViewReadsTheOriginalRegion() {
        ComplexMatrixD A = Matrices.randomUniformComplexD(9, 8, SEED);
        ComplexMatrixD V = Matrices.view(A, 1, 2, 7, 6);
        ComplexMatrixD W = Matrices.view(V, 2, 1, 4, 3);
        assertTrue(W instanceof ComplexMatrixDView);
        assertRegion(A, new int[] { 3, 3, 5, 5 }, W);
        A.set(4, 4, 99.0, -99.0);
        assertBitsZ("live through a view of a view", new ZdImpl(99.0, -99.0), W.get(1, 1));
    }

    @Test
    public void testIllegalRegionsAreRejected() {
        ComplexMatrixD A = Matrices.randomUniformComplexD(4, 3, SEED);
        try {
            Matrices.view((ComplexMatrixD) null, 0, 0, 0, 0);
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
            ComplexMatrixD A = Matrices.randomUniformComplexD(rows, cols, SEED + 100 * rows + cols);
            for (int draw = 0; draw < 20; ++draw) {
                int[] reg = region(rnd, rows, cols);
                ComplexMatrixD V = A.view(reg[0], reg[1], reg[2], reg[3]);
                ComplexMatrixD R = A.viewRows(reg[0], reg[2]);
                ComplexMatrixD C = A.viewColumns(reg[1], reg[3]);
                assertRegion(A, reg, V);
                assertRegion(A, new int[] { reg[0], 0, reg[2], cols - 1 }, R);
                assertRegion(A, new int[] { 0, reg[1], rows - 1, reg[3] }, C);
                assertBitsArray("viewRows", A.selectSubmatrix(reg[0], 0, reg[2], cols - 1).getArrayUnsafe(),
                        R.copy().getArrayUnsafe());
                assertBitsArray("viewColumns", A.selectConsecutiveColumns(reg[1], reg[3]).getArrayUnsafe(),
                        C.copy().getArrayUnsafe());
                assertTrue(V instanceof ComplexMatrixDView && R instanceof ComplexMatrixDView
                        && C instanceof ComplexMatrixDView);
            }
            ComplexMatrixD V = A.view(0, 0, rows - 1, cols - 1);
            ComplexMatrixD R = A.viewRows(rows - 1, rows - 1);
            ComplexMatrixD C = A.viewColumns(cols - 1, cols - 1);
            A.set(rows - 1, cols - 1, 12345.0, -1.0);
            Zd want = new ZdImpl(12345.0, -1.0);
            assertBitsZ("live view", want, V.get(rows - 1, cols - 1));
            assertBitsZ("live viewRows", want, R.get(0, cols - 1));
            assertBitsZ("live viewColumns", want, C.get(rows - 1, 0));
            for (ComplexMatrixD M : new ComplexMatrixD[] { V, R, C }) {
                try {
                    M.set(0, 0, 1.0, 1.0);
                    fail("set on a view");
                } catch (UnsupportedOperationException expected) {
                }
            }
        }
    }

    @Test
    public void testViewMethodsOnAViewReadTheOriginalParent() {
        ComplexMatrixD P = Matrices.randomUniformComplexD(10, 11, SEED);
        ComplexMatrixD V = Matrices.view(P, 2, 3, 7, 8);
        ComplexMatrixD R = V.viewRows(1, 2);
        ComplexMatrixD C = V.viewColumns(1, 4);
        ComplexMatrixD W = V.view(1, 1, 3, 2);
        assertTrue(R instanceof ComplexMatrixDView && C instanceof ComplexMatrixDView
                && W instanceof ComplexMatrixDView);
        assertRegion(P, new int[] { 3, 3, 4, 8 }, R);
        assertRegion(P, new int[] { 2, 4, 7, 7 }, C);
        assertRegion(P, new int[] { 3, 4, 5, 5 }, W);
        P.set(3, 4, 99.0, 0.5);
        Zd want = new ZdImpl(99.0, 0.5);
        assertBitsZ("live viewRows", want, R.get(0, 1));
        assertBitsZ("live viewColumns", want, C.get(1, 0));
        assertBitsZ("live view", want, W.get(0, 0));
    }

    @Test
    public void testViewMethodsRejectIllegalRanges() {
        ComplexMatrixD A = Matrices.randomUniformComplexD(4, 3, SEED);
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
        ComplexMatrixD A = Matrices.randomUniformComplexD(6, 6, SEED);
        ComplexMatrixD V = Matrices.view(A, 1, 1, 3, 3);
        assertEquals(3, V.numRows());
        assertEquals(3, V.numColumns());
        for (int[] ix : new int[][] { { 3, 0 }, { 0, 3 }, { -1, 0 }, { 0, -1 } }) {
            assertEquals("get " + ix[0] + ", " + ix[1], IllegalArgumentException.class,
                    exception(() -> V.get(ix[0], ix[1])));
            assertEquals("get into " + ix[0] + ", " + ix[1], IllegalArgumentException.class,
                    exception(() -> V.get(ix[0], ix[1], new ZdImpl(0.0))));
        }
        ComplexMatrixD C = V.copy();
        assertEquals(exception(() -> C.get(0, 0, null)), exception(() -> V.get(0, 0, null)));
    }

    @Test
    public void testToScalar() {
        ComplexMatrixD A = Matrices.randomUniformComplexD(3, 3, SEED);
        assertBitsZ("scalar", A.get(2, 1), Matrices.view(A, 2, 1, 2, 1).toScalar());
        try {
            Matrices.view(A, 0, 0, 1, 0).toScalar();
            fail("not a scalar");
        } catch (IllegalStateException expected) {
        }
    }

    @Test
    public void testToStringMatchesTheCopy() {
        ComplexMatrixD A = Matrices.randomUniformComplexD(5, 4, SEED);
        ComplexMatrixD V = Matrices.view(A, 1, 0, 3, 2);
        assertEquals(V.copy().toString(), V.toString());
    }

    @Test
    public void testEveryMutatorThrowsAndLeavesEverythingUntouched() {
        RefusalFixture f = new RefusalFixture();
        List<Refusal> refusals = f.refusals();
        assertEquals(18, refusals.size());
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
        ComplexMatrixD V = Matrices.view(Matrices.randomUniformComplexD(6, 5, SEED), 1, 1, 4, 3);
        List<Refusal> bad = new ArrayList<>();
        bad.add(new Refusal("set out of range", m -> m.set(99, 99, 1.0, 1.0)));
        bad.add(new Refusal("setUnsafe out of range", m -> m.setUnsafe(-1, -1, 1.0, 1.0)));
        bad.add(new Refusal("addInplace null", m -> m.addInplace(null)));
        bad.add(new Refusal("setInplace null", m -> m.setInplace(null)));
        bad.add(new Refusal("setSubmatrixInplace null", m -> m.setSubmatrixInplace(0, 0, null, 0, 0, 0, 0)));
        bad.add(new Refusal("zeroizeSubEpsilonInplace illegal k", m -> m.zeroizeSubEpsilonInplace(0)));
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
        for (boolean anchored : new boolean[] { false, true }) {
            ReadFixture f = new ReadFixture(anchored);
            List<Read> reads = f.reads();
            assertEquals(59, reads.size());
            Set<String> names = new HashSet<>();
            for (Read r : reads) {
                String what = r.name + (anchored ? " at (0, 0)" : " with offset");
                assertTrue("duplicate " + r.name, names.add(r.name));
                ComplexMatrixD view = r.square ? f.Q : f.V;
                ComplexMatrixD expected = r.square ? f.EQ : f.E;
                Object got = r.call.apply(view);
                Object want = r.call.apply(expected);
                assertAgree(what, r.exact, want, got);
                if (got instanceof ComplexMatrixD) {
                    assertTrue(what + " returned a view", !(got instanceof ComplexMatrixDView));
                    ((ComplexMatrixD) got).set(0, 0, 12345.0, 0.0);
                } else if (got instanceof MatrixD) {
                    ((MatrixD) got).set(0, 0, 12345.0);
                }
                f.assertParentsUntouched(what);
            }
            f.assertArgumentsUntouched();
        }
    }

    @Test
    public void testAnchoredViewsReadAParentWithMoreRows() {
        // a view at (0, 0) is read in place with the parent's row count as leading dimension
        ComplexMatrixD P = Matrices.randomUniformComplexD(12, 9, SEED);
        ComplexMatrixD B = Matrices.randomUniformComplexD(4, 3, SEED + 1L);
        for (int[] reg : new int[][] { { 0, 0, 4, 3 }, { 0, 0, 11, 3 }, { 0, 0, 0, 3 }, { 1, 0, 5, 3 },
                { 0, 5, 4, 8 } }) {
            ComplexMatrixD V = Matrices.view(P, reg[0], reg[1], reg[2], reg[3]);
            ComplexMatrixD E = V.copy();
            String what = "region " + Arrays.toString(reg);
            assertClose(what + " times", E.times(B), V.times(B));
            assertClose(what + " conjugateTransposedTimes", E.conjugateTransposedTimes(), V.conjugateTransposedTimes());
            assertClose(what + " timesConjugateTransposed", E.timesConjugateTransposed(), V.timesConjugateTransposed());
        }
    }

    @Test
    public void testCopyReadsFollowTheLiveParent() {
        ComplexMatrixD G = Matrices.randomUniformComplexD(9, 8, SEED);
        ComplexMatrixD V = Matrices.view(G, 1, 2, 6, 6);
        ComplexMatrixD B = Matrices.randomUniformComplexD(6, 5, SEED + 1L);
        double before = V.normF();
        G.set(3, 4, 100.0, -100.0);
        ComplexMatrixD E = G.selectSubmatrix(1, 2, 6, 6);
        assertBits("normF after a parent write", E.normF(), V.normF());
        assertTrue("normF must change", Double.doubleToRawLongBits(before) != Double
                .doubleToRawLongBits(V.normF()));
        assertBitsArray("plus after a parent write", E.plus(B).getArrayUnsafe(), V.plus(B).getArrayUnsafe());
    }

    @Test
    public void testCopyReadsReportShapeErrorsLikeTheCopy() {
        ComplexMatrixD G = Matrices.randomUniformComplexD(9, 8, SEED);
        ComplexMatrixD V = Matrices.view(G, 1, 2, 6, 6);
        ComplexMatrixD E = G.selectSubmatrix(1, 2, 6, 6);
        ComplexMatrixD wrong = Matrices.randomUniformComplexD(6, 4, SEED);
        List<Read> bad = new ArrayList<>();
        bad.add(new Read("plus wrong shape", true, false, m -> m.plus(wrong)));
        bad.add(new Read("times wrong shape", true, false, m -> m.times(wrong)));
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
            ComplexMatrixD A = Matrices.randomUniformComplexD(s[0], s[1], rnd.nextLong());
            for (int draw = 0; draw < 40; ++draw) {
                int[] reg = region(rnd, s[0], s[1]);
                ComplexMatrixD V = Matrices.view(A, reg[0], reg[1], reg[2], reg[3]);
                int[] sub = region(rnd, V.numRows(), V.numColumns());
                int h = sub[2] - sub[0] + 1;
                int w = sub[3] - sub[1] + 1;
                int extraR = rnd.nextInt(4);
                int extraC = rnd.nextInt(4);
                checkSubmatrix("draw " + draw, V, sub, Matrices.randomUniformComplexD(h + extraR, w + extraC,
                        rnd.nextLong()), rnd.nextInt(extraR + 1), rnd.nextInt(extraC + 1));
            }
        }
        // every block height on both sides of the threshold
        ComplexMatrixD A = Matrices.randomUniformComplexD(48, 5, SEED);
        ComplexMatrixD V = Matrices.view(A, 0, 1, 47, 4);
        for (int len = 1; len <= 47; ++len) {
            checkSubmatrix("len " + len, V, new int[] { 1, 0, len, 3 },
                    Matrices.randomUniformComplexD(48, 4, SEED + len), 48 - len, 0);
        }
    }

    @Test
    public void testSubmatrixChecksTheViewBounds() {
        ComplexMatrixD A = Matrices.randomUniformComplexD(8, 8, SEED);
        ComplexMatrixD V = Matrices.view(A, 2, 2, 5, 5);
        ComplexMatrixD C = V.copy();
        List<Read> bad = new ArrayList<>();
        bad.add(new Read("region outside the view", true, false,
                m -> m.submatrix(0, 0, 4, 4, Matrices.createComplexD(8, 8), 0, 0)));
        bad.add(new Read("target too small", true, false,
                m -> m.submatrix(0, 0, 3, 3, Matrices.createComplexD(2, 2), 0, 0)));
        bad.add(new Read("target position out of range", true, false,
                m -> m.submatrix(0, 0, 1, 1, Matrices.createComplexD(4, 4), 3, 0)));
        for (Read r : bad) {
            Class<?> want = thrown(r, C);
            assertTrue(r.name + " must throw on the copy", want != null);
            assertEquals(r.name, want, thrown(r, V));
        }
    }

    @Test
    public void testDecompositionsAgreeWithTheCopy() {
        ComplexMatrixD T = Matrices.randomUniformComplexD(10, 9, SEED);
        ComplexMatrixD S = Matrices.randomUniformComplexD(9, 9, SEED + 1L).addInplace(20.0, 0.0,
                Matrices.identityComplexD(9));
        double[] t0 = T.getArrayUnsafe().clone();
        double[] s0 = S.getArrayUnsafe().clone();
        ComplexMatrixD V = Matrices.view(T, 2, 3, 8, 7);
        ComplexMatrixD C = V.copy();
        ComplexMatrixD Q = Matrices.view(S, 2, 2, 7, 7);
        ComplexMatrixD CQ = Q.copy();

        for (boolean full : new boolean[] { true, false }) {
            String what = "svd(" + full + ")";
            SvdComplexD want = C.svd(full);
            SvdComplexD got = V.svd(full);
            assertValues(what + " S", false, want.getS(), got.getS());
            assertEquals(what, want.hasSingularVectors(), got.hasSingularVectors());
            if (want.hasSingularVectors()) {
                assertClose(what + " U", want.getU(), got.getU());
                assertClose(what + " Vh", want.getVh(), got.getVh());
            }
        }
        SvdEconComplexD econWant = C.svdEcon();
        SvdEconComplexD econGot = V.svdEcon();
        assertValues("svdEcon S", false, econWant.getS(), econGot.getS());
        assertClose("svdEcon U", econWant.getU(), econGot.getU());
        assertClose("svdEcon Vh", econWant.getVh(), econGot.getVh());

        QrdComplexD qrWant = C.qrd();
        QrdComplexD qrGot = V.qrd();
        assertClose("qrd Q", qrWant.getQ(), qrGot.getQ());
        assertClose("qrd R", qrWant.getR(), qrGot.getR());

        assertLud("lud tall", C.lud(), V.lud());
        assertLud("lud square", CQ.lud(), Q.lud());

        assertValues("singularValues", false, C.singularValues(), V.singularValues());
        assertValue("norm2", false, C.norm2(), V.norm2());

        for (boolean full : new boolean[] { true, false }) {
            String what = "evd(" + full + ")";
            EvdComplexD want = CQ.evd(full);
            EvdComplexD got = Q.evd(full);
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
        ComplexMatrixD T = Matrices.randomUniformComplexD(10, 9, SEED);
        ComplexMatrixD S = Matrices.randomUniformComplexD(9, 9, SEED + 1L);
        ComplexMatrixD V = Matrices.view(T, 2, 3, 8, 7);
        ComplexMatrixD Q = Matrices.view(S, 0, 0, 5, 5);
        SvdEconComplexD svd = V.svdEcon();
        QrdComplexD qr = V.qrd();
        EvdComplexD evd = Q.evd(true);
        double[] u = svd.getU().getArrayUnsafe().clone();
        double[] sv = svd.getS().clone();
        double[] vh = svd.getVh().getArrayUnsafe().clone();
        double[] q = qr.getQ().getArrayUnsafe().clone();
        double[] r = qr.getR().getArrayUnsafe().clone();
        double[] ev = evd.getEigenvectors().getArrayUnsafe().clone();
        Zd lambda = evd.getEigenvalues()[0].copy();
        T.set(4, 5, 1000.0, 1000.0);
        S.set(1, 1, 1000.0, 1000.0);
        assertBitsArray("svdEcon U", u, svd.getU().getArrayUnsafe());
        assertBitsArray("svdEcon S", sv, svd.getS());
        assertBitsArray("svdEcon Vh", vh, svd.getVh().getArrayUnsafe());
        assertBitsArray("qrd Q", q, qr.getQ().getArrayUnsafe());
        assertBitsArray("qrd R", r, qr.getR().getArrayUnsafe());
        assertBitsArray("evd vectors", ev, evd.getEigenvectors().getArrayUnsafe());
        assertBitsZ("evd value", lambda, evd.getEigenvalues()[0]);
    }

    @Test
    public void testEvdRejectsANonSquareView() {
        ComplexMatrixD T = Matrices.randomUniformComplexD(10, 9, SEED);
        ComplexMatrixD V = Matrices.view(T, 2, 3, 8, 7);
        Read evd = new Read("evd on a non-square view", false, false, m -> m.evd(true));
        assertEquals(IllegalArgumentException.class, thrown(evd, V.copy()));
        assertEquals(IllegalArgumentException.class, thrown(evd, V));
    }

    @Test
    public void testEveryInterfaceMethodIsClassified() throws Exception {
        assertEquals(18, REFUSED.size());
        assertEquals(34, COPY_READS.size());
        assertEquals(25, GEMM_READS.size());
        assertEquals(14, DIRECT_READS.size());
        assertEquals(17, DIMENSIONS.size());
        assertEquals(3, INHERITED.size());
        assertEquals(new RefusalFixture().refusals().size(), REFUSED.size());
        assertEquals(new ReadFixture(false).reads().size(), COPY_READS.size() + GEMM_READS.size());

        Set<String> all = new TreeSet<>();
        for (Set<String> s : Arrays.asList(REFUSED, COPY_READS, GEMM_READS, DIRECT_READS, DIMENSIONS, INHERITED)) {
            for (String sig : s) {
                assertTrue("classified twice: " + sig, all.add(sig));
            }
        }
        Set<String> declared = new TreeSet<>();
        for (Method m : ComplexMatrixD.class.getMethods()) {
            String sig = sig(m);
            declared.add(sig);
            assertEquals("default method " + sig, INHERITED.contains(sig), m.isDefault());
            assertTrue("not classified: " + sig, all.contains(sig));
            Class<?> owner = ComplexMatrixDView.class.getMethod(m.getName(), m.getParameterTypes())
                    .getDeclaringClass();
            Class<?> want = DIMENSIONS.contains(sig) ? DimensionsBase.class
                    : INHERITED.contains(sig) ? ComplexMatrixDConduct.class : ComplexMatrixDView.class;
            assertEquals("declaring class of " + sig, want, owner);
            if (INHERITED.contains(sig)) {
                Class<?> simple = SimpleComplexMatrixD.class.getMethod(m.getName(), m.getParameterTypes())
                        .getDeclaringClass();
                assertEquals("declaring class in SimpleComplexMatrixD of " + sig, ComplexMatrixDConduct.class,
                        simple);
            }
        }
        for (String sig : all) {
            assertTrue("not in the interface: " + sig, declared.contains(sig));
        }
        assertEquals(111, declared.size());

        ComplexMatrixD V = Matrices.view(Matrices.randomUniformComplexD(6, 5, SEED), 1, 1, 4, 3);
        for (Method m : ComplexMatrixD.class.getMethods()) {
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
        ComplexMatrixD P = Matrices.randomUniformComplexD(50, 9, SEED);
        double[] p0 = P.getArrayUnsafe().clone();
        ComplexMatrixD block = Matrices.view(P, 2, 1, 46, 5);
        ComplexMatrixD anchored = Matrices.view(P, 0, 0, 44, 4);
        ComplexMatrixD column = Matrices.view(P, 2, 6, 46, 6);
        Supplier<ComplexMatrixD> t455 = () -> Matrices.randomUniformComplexD(45, 5, SEED + 1L);
        Supplier<ComplexMatrixD> t745 = () -> Matrices.randomUniformComplexD(7, 45, SEED + 2L);
        Supplier<ComplexMatrixD> square = () -> Matrices.randomUniformComplexD(45, 45, SEED + 3L).addInplace(50.0,
                0.0, Matrices.identityComplexD(45));

        List<ArgCase> cases = new ArrayList<>();
        for (ComplexMatrixD b : new ComplexMatrixD[] { block, anchored }) {
            String at = b == anchored ? " at (0, 0)" : "";
            cases.add(new ArgCase("addInplace" + at, true, b, t455, (t, x) -> t.addInplace(x)));
            cases.add(new ArgCase("plus" + at, true, b, t455, (t, x) -> t.plus(x)));
            cases.add(new ArgCase("times" + at, false, b, t745, (t, x) -> t.times(x)));
            cases.add(new ArgCase("setInplace" + at, true, b, t455, (t, x) -> t.setInplace(x)));
            cases.add(new ArgCase("setSubmatrixInplace height 10" + at, true, b, t455,
                    (t, x) -> t.setSubmatrixInplace(0, 0, x, 0, 0, 9, 4)));
            cases.add(new ArgCase("setSubmatrixInplace height 45" + at, true, b, t455,
                    (t, x) -> t.setSubmatrixInplace(0, 0, x, 0, 0, 44, 4)));
            cases.add(new ArgCase("hadamard" + at, true, b, t455, (t, x) -> t.hadamard(x)));
            cases.add(new ArgCase("Matrices.distance" + at, true, b, t455, (t, x) -> Matrices.distance(t, x)));
            cases.add(new ArgCase("Matrices.sumColumns" + at, true, b, t455, (t, x) -> Matrices.sumColumns(x)));
            cases.add(new ArgCase("solve" + at, false, b, square,
                    (t, x) -> t.solve(x, Matrices.createComplexD(45, 5))));
        }
        cases.add(new ArgCase("appendColumn", true, column, t455, (t, x) -> t.appendColumn(x)));

        for (ArgCase c : cases) {
            ComplexMatrixD t1 = c.target.get();
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
        final ComplexMatrixD arg;
        final Supplier<ComplexMatrixD> target;
        final BiFunction<ComplexMatrixD, ComplexMatrixD, Object> call;

        ArgCase(String name, boolean exact, ComplexMatrixD arg, Supplier<ComplexMatrixD> target,
                BiFunction<ComplexMatrixD, ComplexMatrixD, Object> call) {
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
        final Consumer<ComplexMatrixD> call;

        Refusal(String name, Consumer<ComplexMatrixD> call) {
            this.name = name;
            this.call = call;
        }
    }

    // a 4 x 3 view on a 6 x 5 parent plus ordinary arguments of matching shape
    static final class RefusalFixture {
        final ComplexMatrixD A = Matrices.randomUniformComplexD(6, 5, SEED);
        final ComplexMatrixD V = Matrices.view(A, 1, 1, 4, 3);
        final ComplexMatrixD B = Matrices.randomUniformComplexD(4, 3, SEED + 1L);
        final ComplexMatrixD col = Matrices.randomUniformComplexD(4, 1, SEED + 2L);
        final double[] a0 = A.getArrayUnsafe().clone();
        final double[] b0 = B.getArrayUnsafe().clone();
        final double[] col0 = col.getArrayUnsafe().clone();

        List<Refusal> refusals() {
            List<Refusal> l = new ArrayList<>();
            l.add(new Refusal("scaleInplace", m -> m.scaleInplace(2.0, 1.0)));
            l.add(new Refusal("addInplace(B)", m -> m.addInplace(B)));
            l.add(new Refusal("addInplace(alpha, B)", m -> m.addInplace(2.0, 1.0, B)));
            l.add(new Refusal("zeroInplace", m -> m.zeroInplace()));
            l.add(new Refusal("setInplace(other)", m -> m.setInplace(B)));
            l.add(new Refusal("setInplace(alpha, other)", m -> m.setInplace(2.0, 1.0, B)));
            l.add(new Refusal("setColumnInplace", m -> m.setColumnInplace(0, col)));
            l.add(new Refusal("setInplaceUpperTrapezoidal", m -> m.setInplaceUpperTrapezoidal(B)));
            l.add(new Refusal("setInplaceLowerTrapezoidal", m -> m.setInplaceLowerTrapezoidal(B)));
            l.add(new Refusal("set", m -> m.set(0, 0, 1.0, 1.0)));
            l.add(new Refusal("add(row, col, valr, vali)", m -> m.add(0, 0, 1.0, 1.0)));
            l.add(new Refusal("setSubmatrixInplace", m -> m.setSubmatrixInplace(0, 0, B, 0, 0, 1, 1)));
            l.add(new Refusal("zeroizeSubEpsilonInplace", m -> m.zeroizeSubEpsilonInplace(1)));
            l.add(new Refusal("zeroizeSubEpsilonRelativeInplace", m -> m.zeroizeSubEpsilonRelativeInplace(1)));
            l.add(new Refusal("sanitizeNonFiniteInplace", m -> m.sanitizeNonFiniteInplace(0.0, 1.0, -1.0)));
            l.add(new Refusal("sanitizeNaNInplace", m -> m.sanitizeNaNInplace(0.0)));
            l.add(new Refusal("setUnsafe", m -> m.setUnsafe(0, 0, 1.0, 1.0)));
            l.add(new Refusal("getArrayUnsafe", m -> m.getArrayUnsafe()));
            return l;
        }

        void assertUntouched(String after) {
            assertBitsArray(after + ": parent", a0, A.getArrayUnsafe());
            assertBitsArray(after + ": B", b0, B.getArrayUnsafe());
            assertBitsArray(after + ": column vector", col0, col.getArrayUnsafe());
        }
    }

    // ---------------------------------------------------------------- reads

    static final class Read {
        final String name;
        final boolean exact;
        final boolean square;
        final Function<ComplexMatrixD, Object> call;

        Read(String name, boolean exact, boolean square, Function<ComplexMatrixD, Object> call) {
            this.name = name;
            this.exact = exact;
            this.square = square;
            this.call = call;
        }
    }

    // a 6 x 5 view on a 9 x 8 parent, a 6 x 6 well conditioned view on a 9 x 9
    // parent, either with an offset or at (0, 0), their materialized regions and
    // shared ordinary arguments
    static final class ReadFixture {
        final ComplexMatrixD G = Matrices.randomUniformComplexD(9, 8, SEED);
        final ComplexMatrixD S = Matrices.randomUniformComplexD(9, 9, SEED + 1L).addInplace(20.0, 0.0,
                Matrices.identityComplexD(9));
        final ComplexMatrixD V;
        final ComplexMatrixD E;
        final ComplexMatrixD Q;
        final ComplexMatrixD EQ;
        final double[] g0 = G.getArrayUnsafe().clone();
        final double[] s0 = S.getArrayUnsafe().clone();

        final List<ComplexMatrixD> args = new ArrayList<>();
        final List<double[]> args0 = new ArrayList<>();
        final ComplexMatrixD b65 = arg(6, 5);
        final ComplexMatrixD b54 = arg(5, 4);
        final ComplexMatrixD b64 = arg(6, 4);
        final ComplexMatrixD b45 = arg(4, 5);
        final ComplexMatrixD b46 = arg(4, 6);
        final ComplexMatrixD b43 = arg(4, 3);
        final ComplexMatrixD b32 = arg(3, 2);
        final ComplexMatrixD b62 = arg(6, 2);
        final ComplexMatrixD col6 = arg(6, 1);
        final ComplexMatrixD d66 = arg(6, 6).addInplace(20.0, 0.0, Matrices.identityComplexD(6));
        final ComplexMatrixD c64 = arg(6, 4);
        final ComplexMatrixD c54 = arg(5, 4);
        final MatrixD r54 = Matrices.randomUniformD(5, 4, SEED + 99L);
        final double[] r540 = r54.getArrayUnsafe().clone();

        ComplexMatrixD arg(int rows, int cols) {
            ComplexMatrixD m = Matrices.randomUniformComplexD(rows, cols, SEED + 10L + args.size());
            args.add(m);
            return m;
        }

        // runs after the field initializers, so d66 is already shifted
        ReadFixture(boolean anchored) {
            int off = anchored ? 0 : 1;
            V = Matrices.view(G, off, 2 * off, off + 5, 2 * off + 4);
            E = G.selectSubmatrix(off, 2 * off, off + 5, 2 * off + 4);
            Q = Matrices.view(S, 2 * off, 2 * off, 2 * off + 5, 2 * off + 5);
            EQ = S.selectSubmatrix(2 * off, 2 * off, 2 * off + 5, 2 * off + 5);
            for (ComplexMatrixD m : args) {
                args0.add(m.getArrayUnsafe().clone());
            }
        }

        List<Read> reads() {
            List<Read> l = new ArrayList<>();
            l.add(new Read("scale(alpha, B)", true, false, m -> m.scale(2.5, -1.5, out(6, 5))));
            l.add(new Read("conjTrans", true, false, m -> m.conjTrans(out(5, 6))));
            l.add(new Read("trans", true, false, m -> m.trans(out(5, 6))));
            l.add(new Read("add(B, C)", true, false, m -> m.add(b65, out(6, 5))));
            l.add(new Read("add(alpha, B, C)", true, false, m -> m.add(2.5, -1.5, b65, out(6, 5))));
            l.add(new Read("mult(B, C)", false, false, m -> m.mult(b54, out(6, 4))));
            l.add(new Read("mult(alpha, B, C)", false, false, m -> m.mult(2.5, -1.5, b54, out(6, 4))));
            l.add(new Read("multAdd(B, C)", false, false, m -> m.multAdd(b54, c64.copy())));
            l.add(new Read("multAdd(alpha, B, C)", false, false, m -> m.multAdd(2.5, -1.5, b54, c64.copy())));
            l.add(new Read("conjTransABmult(B, C)", false, false, m -> m.conjTransABmult(b46, out(5, 4))));
            l.add(new Read("conjTransABmult(alpha, B, C)", false, false,
                    m -> m.conjTransABmult(2.5, -1.5, b46, out(5, 4))));
            l.add(new Read("conjTransAmult(B, C)", false, false, m -> m.conjTransAmult(b64, out(5, 4))));
            l.add(new Read("conjTransAmult(alpha, B, C)", false, false,
                    m -> m.conjTransAmult(2.5, -1.5, b64, out(5, 4))));
            l.add(new Read("conjTransBmult(B, C)", false, false, m -> m.conjTransBmult(b45, out(6, 4))));
            l.add(new Read("conjTransBmult(alpha, B, C)", false, false,
                    m -> m.conjTransBmult(2.5, -1.5, b45, out(6, 4))));
            l.add(new Read("conjTransABmultAdd(B, C)", false, false, m -> m.conjTransABmultAdd(b46, c54.copy())));
            l.add(new Read("conjTransABmultAdd(alpha, B, C)", false, false,
                    m -> m.conjTransABmultAdd(2.5, -1.5, b46, c54.copy())));
            l.add(new Read("conjTransAmultAdd(B, C)", false, false, m -> m.conjTransAmultAdd(b64, c54.copy())));
            l.add(new Read("conjTransAmultAdd(alpha, B, C)", false, false,
                    m -> m.conjTransAmultAdd(2.5, -1.5, b64, c54.copy())));
            l.add(new Read("conjTransBmultAdd(B, C)", false, false, m -> m.conjTransBmultAdd(b45, c64.copy())));
            l.add(new Read("conjTransBmultAdd(alpha, B, C)", false, false,
                    m -> m.conjTransBmultAdd(2.5, -1.5, b45, c64.copy())));
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
            l.add(new Read("times(ComplexMatrixD)", false, false, m -> m.times(b54)));
            l.add(new Read("timesTimes", false, false, m -> m.timesTimes(b54, b43)));
            l.add(new Read("timesMany", false, false, m -> m.timesMany(b54, b43, b32)));
            l.add(new Read("timesConjugateTransposed()", false, false, m -> m.timesConjugateTransposed()));
            l.add(new Read("timesConjugateTransposed(B)", false, false, m -> m.timesConjugateTransposed(b45)));
            l.add(new Read("conjugateTransposedTimes()", false, false, m -> m.conjugateTransposedTimes()));
            l.add(new Read("conjugateTransposedTimes(B)", false, false, m -> m.conjugateTransposedTimes(b64)));
            l.add(new Read("times(MatrixD)", false, false, m -> m.times(r54)));
            l.add(new Read("plus", true, false, m -> m.plus(b65)));
            l.add(new Read("timesPlus", false, false, m -> m.timesPlus(b54, c64)));
            l.add(new Read("timesMinus", false, false, m -> m.timesMinus(b54, c64)));
            l.add(new Read("minus", true, false, m -> m.minus(b65)));
            l.add(new Read("uminus", true, false, m -> m.uminus()));
            l.add(new Read("abs", true, false, m -> m.abs()));
            l.add(new Read("conjugateTranspose", true, false, m -> m.conjugateTranspose()));
            l.add(new Read("transpose", true, false, m -> m.transpose()));
            l.add(new Read("inverse", false, true, m -> m.inverse()));
            l.add(new Read("hadamard(B)", true, false, m -> m.hadamard(b65)));
            l.add(new Read("reshape", true, false, m -> m.reshape(5, 6)));
            l.add(new Read("toRealMatrix", true, false, m -> m.toRealMatrix()));
            return l;
        }

        static ComplexMatrixD out(int rows, int cols) {
            return Matrices.createComplexD(rows, cols);
        }

        void assertParentsUntouched(String after) {
            assertBitsArray(after + ": parent G", g0, G.getArrayUnsafe());
            assertBitsArray(after + ": parent S", s0, S.getArrayUnsafe());
        }

        void assertArgumentsUntouched() {
            for (int i = 0; i < args.size(); ++i) {
                assertBitsArray("argument " + i, args0.get(i), args.get(i).getArrayUnsafe());
            }
            assertBitsArray("real argument", r540, r54.getArrayUnsafe());
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

    static Class<?> thrown(Read r, ComplexMatrixD m) {
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
        } else if (want instanceof Zd) {
            assertValue(what + " re", exact, ((Zd) want).re(), ((Zd) got).re());
            assertValue(what + " im", exact, ((Zd) want).im(), ((Zd) got).im());
        } else if (want instanceof double[][]) {
            double[][] w = (double[][]) want;
            double[][] g = (double[][]) got;
            assertEquals(what + ": rows", w.length, g.length);
            for (int i = 0; i < w.length; ++i) {
                assertValues(what + " row " + i, exact, w[i], g[i]);
            }
        } else if (want instanceof ComplexMatrixD) {
            ComplexMatrixD w = (ComplexMatrixD) want;
            ComplexMatrixD g = (ComplexMatrixD) got;
            assertEquals(what + ": rows", w.numRows(), g.numRows());
            assertEquals(what + ": cols", w.numColumns(), g.numColumns());
            assertValues(what, exact, w.getArrayUnsafe(), g.getArrayUnsafe());
        } else if (want instanceof MatrixD) {
            MatrixD w = (MatrixD) want;
            MatrixD g = (MatrixD) got;
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
    static void checkSubmatrix(String what, ComplexMatrixD V, int[] sub, ComplexMatrixD target, int rb, int cb) {
        ComplexMatrixD B1 = target.copy();
        ComplexMatrixD B2 = target.copy();
        ComplexMatrixD got = V.submatrix(sub[0], sub[1], sub[2], sub[3], B1, rb, cb);
        assertTrue(what + ": must return the target", got == B1);
        V.copy().submatrix(sub[0], sub[1], sub[2], sub[3], B2, rb, cb);
        assertBitsArray(what, B2.getArrayUnsafe(), B1.getArrayUnsafe());
    }

    static void assertClose(String what, ComplexMatrixD want, ComplexMatrixD got) {
        assertEquals(what + ": rows", want.numRows(), got.numRows());
        assertEquals(what + ": cols", want.numColumns(), got.numColumns());
        assertValues(what, false, want.getArrayUnsafe(), got.getArrayUnsafe());
    }

    static void assertLud(String what, LudComplexD want, LudComplexD got) {
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
    static void assertRegion(ComplexMatrixD A, int[] reg, ComplexMatrixD M) {
        double[] a = A.getArrayUnsafe();
        int rows = reg[2] - reg[0] + 1;
        int cols = reg[3] - reg[1] + 1;
        assertEquals(rows, M.numRows());
        assertEquals(cols, M.numColumns());
        Zd out = new ZdImpl(0.0);
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                int idx = 2 * ((reg[1] + col) * A.numRows() + reg[0] + row);
                Zd want = new ZdImpl(a[idx], a[idx + 1]);
                String at = "(" + row + ", " + col + ")";
                assertBitsZ(at, want, M.get(row, col));
                assertBitsZ("unsafe " + at, want, M.getUnsafe(row, col));
                M.get(row, col, out);
                assertBitsZ("into " + at, want, out);
                M.getUnsafe(row, col, out);
                assertBitsZ("unsafe into " + at, want, out);
            }
        }
    }

    static void assertBitsZ(String what, Zd want, Zd got) {
        assertBits(what + " re", want.re(), got.re());
        assertBits(what + " im", want.im(), got.im());
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
