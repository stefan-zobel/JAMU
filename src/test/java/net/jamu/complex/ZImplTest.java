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
package net.jamu.complex;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Tests for the mutable complex implementations.
 */
public final class ZImplTest {

    /** the first exponent, the last one and the step of the test ensemble */
    private static final int MIN_EXP_D = -300;
    private static final int MAX_EXP_D = 300;
    private static final int MIN_EXP_F = -30;
    private static final int MAX_EXP_F = 30;
    private static final int ANGLES = 8;

    private static double angle(int k) {
        // 0.3 keeps the components off the axes
        return k * Math.PI / 4.0 + 0.3;
    }

    private static double relative(Zd got, Zd want) {
        return new ZdImpl(got.re() - want.re(), got.im() - want.im()).abs() / want.abs();
    }

    @Test
    public void testInvAgreesWithDivDouble() {
        for (int e = MIN_EXP_D; e <= MAX_EXP_D; ++e) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                double re = r * Math.cos(angle(k));
                double im = r * Math.sin(angle(k));
                Zd got = new ZdImpl(re, im).inv();
                Zd want = new ZdImpl(1.0, 0.0).div(new ZdImpl(re, im));
                assertEquals("re at 1e" + e, want.re(), got.re(), 0.0);
                assertEquals("im at 1e" + e, want.im(), got.im(), 0.0);
            }
        }
    }

    @Test
    public void testInvAgreesWithDivFloat() {
        for (int e = MIN_EXP_F; e <= MAX_EXP_F; ++e) {
            float r = (float) Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                float re = (float) (r * Math.cos(angle(k)));
                float im = (float) (r * Math.sin(angle(k)));
                Zf got = new ZfImpl(re, im).inv();
                Zf want = new ZfImpl(1.0f, 0.0f).div(new ZfImpl(re, im));
                assertEquals("re at 1e" + e, want.re(), got.re(), 0.0f);
                assertEquals("im at 1e" + e, want.im(), got.im(), 0.0f);
            }
        }
    }

    @Test
    public void testInvRoundTrip() {
        for (int e = MIN_EXP_D; e <= MAX_EXP_D; ++e) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                Zd z = new ZdImpl(r * Math.cos(angle(k)), r * Math.sin(angle(k)));
                assertTrue("1e" + e, relative(z.copy().inv().inv(), z) <= 1.0e-15);
            }
        }
    }

    @Test
    public void testTimesInverseIsOne() {
        for (int e = MIN_EXP_D; e <= MAX_EXP_D; ++e) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                Zd z = new ZdImpl(r * Math.cos(angle(k)), r * Math.sin(angle(k)));
                assertTrue("1e" + e, relative(z.copy().mul(z.copy().inv()), Zd.One()) <= 1.0e-15);
            }
        }
    }

    @Test
    public void testInvWhereTheSquaredModulusWouldOverflow() {
        Zf f = new ZfImpl(1.0e20f, 1.0e20f).inv();
        assertEquals("float re", 5.0e-21f, f.re(), 1.0e-26f);
        assertEquals("float im", -5.0e-21f, f.im(), 1.0e-26f);
        Zd d = new ZdImpl(1.0e200, 1.0e200).inv();
        assertEquals("double re", 5.0e-201, d.re(), 1.0e-215);
        assertEquals("double im", -5.0e-201, d.im(), 1.0e-215);
        Zd u = new ZdImpl(1.0e-200, 1.0e-200).inv();
        assertEquals("double underflow re", 5.0e+199, u.re(), 1.0e+185);
        assertEquals("double underflow im", -5.0e+199, u.im(), 1.0e+185);
    }

    /** e^(re + im i) through the finite branch only, scaled back up */
    private static double[] expReference(double re, double im) {
        Zd w = new ZdImpl(re - 300.0, im).exp();
        double f = Math.exp(300.0);
        return new double[] { w.re() * f, w.im() * f };
    }

    private static boolean usable(double x) {
        return !Double.isInfinite(x) && !Double.isNaN(x) && x != 0.0;
    }

    @Test
    public void testExpOverflowKeepsAnExactZero() {
        Zd d = new ZdImpl(1000.0, 0.0).exp();
        assertEquals(Double.POSITIVE_INFINITY, d.re(), 0.0);
        assertEquals("inf times an exact zero must stay zero", 0.0, d.im(), 0.0);
        Zf f = new ZfImpl(1000.0f, 0.0f).exp();
        assertEquals(Float.POSITIVE_INFINITY, f.re(), 0.0f);
        assertEquals("inf times an exact zero must stay zero", 0.0f, f.im(), 0.0f);
        Zd u = new ZdImpl(-1000.0, 0.0).exp();
        assertEquals(0.0, u.re(), 0.0);
        assertEquals(0.0, u.im(), 0.0);
    }

    @Test
    public void testExpInTheOverflowBand() {
        // e^re overflows here, the product with cos or sin need not
        for (double re : new double[] { 709.9, 710.0, 712.0, 715.0, 720.0, 740.0 }) {
            for (double im : new double[] { 1.0, 1.5, Math.PI / 2.0 }) {
                Zd got = new ZdImpl(re, im).exp();
                double[] want = expReference(re, im);
                if (usable(want[0])) {
                    assertTrue("re at " + re + ", " + im, usable(got.re()));
                    assertTrue("re at " + re + ", " + im, Math.abs(got.re() - want[0]) <= 1.0e-14 * Math.abs(want[0]));
                }
                if (usable(want[1])) {
                    assertTrue("im at " + re + ", " + im, usable(got.im()));
                    assertTrue("im at " + re + ", " + im, Math.abs(got.im() - want[1]) <= 1.0e-14 * Math.abs(want[1]));
                }
            }
        }
    }

    @Test
    public void testExpDoesNotInventFiniteValues() {
        for (double re = 709.79; re <= 800.0; re += 0.37) {
            for (int k = 0; k < 40; ++k) {
                double im = k * 0.157 - 3.0;
                Zd got = new ZdImpl(re, im).exp();
                double[] want = expReference(re, im);
                if (Double.isInfinite(want[0])) {
                    assertTrue("re at " + re, Double.isInfinite(got.re()));
                }
                if (Double.isInfinite(want[1])) {
                    assertTrue("im at " + re, Double.isInfinite(got.im()));
                }
            }
        }
    }

    @Test
    public void testExpBelowOverflowIsUnchanged() {
        for (int r = -700; r <= 700; r += 7) {
            for (int k = 0; k < 12; ++k) {
                double im = k * 0.7 - 4.0;
                Zd got = new ZdImpl(r, im).exp();
                double h = Math.exp(r);
                assertEquals("re at " + r, Double.doubleToLongBits(h * Math.cos(im)),
                        Double.doubleToLongBits(got.re()));
                assertEquals("im at " + r, Double.doubleToLongBits(h * Math.sin(im)),
                        Double.doubleToLongBits(got.im()));
            }
        }
    }

    @Test
    public void testPowOfALargeRealStaysReal() {
        // pow goes through ln().scale().exp(), so it inherits the exp fix
        Zd p = new ZdImpl(1.0e200, 0.0).pow(2.0);
        assertEquals(Double.POSITIVE_INFINITY, p.re(), 0.0);
        assertEquals(0.0, p.im(), 0.0);
    }

    private static void assertZ(String what, double wantRe, double wantIm, Zd got) {
        assertEquals(what + " re", wantRe, got.re(), 0.0);
        assertEquals(what + " im", wantIm, got.im(), 0.0);
    }

    @Test
    public void testPowOfZero() {
        double inf = Double.POSITIVE_INFINITY;
        assertZ("0^2", 0.0, 0.0, new ZdImpl(0.0, 0.0).pow(2.0));
        assertZ("0^0.5", 0.0, 0.0, new ZdImpl(0.0, 0.0).pow(0.5));
        assertZ("0^0", 1.0, 0.0, new ZdImpl(0.0, 0.0).pow(0.0));
        assertZ("0^-2", inf, inf, new ZdImpl(0.0, 0.0).pow(-2.0));
        assertZ("0^NaN", Double.NaN, Double.NaN, new ZdImpl(0.0, 0.0).pow(Double.NaN));
        Zf f = new ZfImpl(0.0f, 0.0f).pow(2.0f);
        assertEquals("float 0^2 re", 0.0f, f.re(), 0.0f);
        assertEquals("float 0^2 im", 0.0f, f.im(), 0.0f);
        Zf g = new ZfImpl(0.0f, 0.0f).pow(0.0f);
        assertEquals("float 0^0 re", 1.0f, g.re(), 0.0f);
        assertEquals("float 0^0 im", 0.0f, g.im(), 0.0f);
    }

    @Test
    public void testPowOfInfinity() {
        double inf = Double.POSITIVE_INFINITY;
        assertZ("inf^2", inf, inf, new ZdImpl(inf, 0.0).pow(2.0));
        assertZ("inf^0", 1.0, 0.0, new ZdImpl(inf, 0.0).pow(0.0));
        assertZ("inf^-2", 0.0, 0.0, new ZdImpl(inf, 0.0).pow(-2.0));
        Zf f = new ZfImpl(Float.POSITIVE_INFINITY, 0.0f).pow(-2.0f);
        assertEquals("float inf^-2 re", 0.0f, f.re(), 0.0f);
        assertEquals("float inf^-2 im", 0.0f, f.im(), 0.0f);
    }

    @Test
    public void testPowOfZeroWithAComplexExponent() {
        double inf = Double.POSITIVE_INFINITY;
        assertZ("0^(2+0i)", 0.0, 0.0, new ZdImpl(0.0, 0.0).pow(new ZdImpl(2.0, 0.0)));
        assertZ("0^(0+0i)", 1.0, 0.0, new ZdImpl(0.0, 0.0).pow(new ZdImpl(0.0, 0.0)));
        assertZ("0^(0+1i)", Double.NaN, Double.NaN, new ZdImpl(0.0, 0.0).pow(new ZdImpl(0.0, 1.0)));
        assertZ("0^(-2+3i)", inf, inf, new ZdImpl(0.0, 0.0).pow(new ZdImpl(-2.0, 3.0)));
    }

    @Test
    public void testPowWithANonFiniteExponent() {
        // only a real exponent is defined for a degenerate base
        double inf = Double.POSITIVE_INFINITY;
        assertZ("0^(2+inf i)", Double.NaN, Double.NaN, new ZdImpl(0.0, 0.0).pow(new ZdImpl(2.0, inf)));
        assertZ("inf^(1+inf i)", Double.NaN, Double.NaN, new ZdImpl(inf, 0.0).pow(new ZdImpl(1.0, inf)));
        assertZ("0^(2+NaN i)", Double.NaN, Double.NaN, new ZdImpl(0.0, 0.0).pow(new ZdImpl(2.0, Double.NaN)));
        Zf f = new ZfImpl(0.0f, 0.0f).pow(new ZfImpl(2.0f, Float.POSITIVE_INFINITY));
        assertEquals("float 0^(2+inf i) re", Float.NaN, f.re(), 0.0f);
        assertEquals("float 0^(2+inf i) im", Float.NaN, f.im(), 0.0f);
        // a real exponent stays real, so both overloads agree
        assertZ("0^(-inf+0i)", inf, inf, new ZdImpl(0.0, 0.0).pow(new ZdImpl(Double.NEGATIVE_INFINITY, 0.0)));
        assertZ("0^-inf", inf, inf, new ZdImpl(0.0, 0.0).pow(Double.NEGATIVE_INFINITY));
    }

    @Test
    public void testPowOfANanBaseStaysNan() {
        assertZ("(inf,NaN)^2", Double.NaN, Double.NaN, new ZdImpl(Double.POSITIVE_INFINITY, Double.NaN).pow(2.0));
        assertZ("(NaN,0)^2", Double.NaN, Double.NaN, new ZdImpl(Double.NaN, 0.0).pow(2.0));
    }

    @Test
    public void testPowOfOrdinaryBasesIsUnchanged() {
        for (int e = -150; e <= 150; e += 3) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                double re = r * Math.cos(angle(k));
                double im = r * Math.sin(angle(k));
                for (double x : new double[] { 2.0, 0.5, -1.5, 0.0 }) {
                    Zd got = new ZdImpl(re, im).pow(x);
                    Zd want = new ZdImpl(re, im).ln().scale(x).exp();
                    assertEquals("re at 1e" + e, Double.doubleToLongBits(want.re()),
                            Double.doubleToLongBits(got.re()));
                    assertEquals("im at 1e" + e, Double.doubleToLongBits(want.im()),
                            Double.doubleToLongBits(got.im()));
                }
            }
        }
    }

    @Test
    public void testInvKeepsTheInfinityConvention() {
        Zd zero = new ZdImpl(0.0, 0.0).inv();
        assertEquals(Double.POSITIVE_INFINITY, zero.re(), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, zero.im(), 0.0);
        Zd inf = new ZdImpl(Double.POSITIVE_INFINITY, 0.0).inv();
        assertEquals(0.0, inf.re(), 0.0);
        assertEquals(0.0, inf.im(), 0.0);
        Zf zeroF = new ZfImpl(0.0f, 0.0f).inv();
        assertEquals(Float.POSITIVE_INFINITY, zeroF.re(), 0.0f);
        assertEquals(Float.POSITIVE_INFINITY, zeroF.im(), 0.0f);
        Zf infF = new ZfImpl(Float.POSITIVE_INFINITY, 0.0f).inv();
        assertEquals(0.0f, infF.re(), 0.0f);
        assertEquals(0.0f, infF.im(), 0.0f);
    }
}
