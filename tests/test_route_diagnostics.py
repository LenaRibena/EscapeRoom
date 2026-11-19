from math import cos, pi, sin
import unittest

from src.route_diagnostics import DiagnosticThresholds, diagnose_routes
from src.route_diagnostics import _curvature_radii
from src.wfc_building import WFCPath
from src.wfc_tiles import PortKind


class DiagnosticBuilding:
    grid_resolution = 1
    routes = ()
    geometries = {}

    def graph_node_to_cell(self, node):
        x, y, floor = node
        return floor, x, y

    def smooth_path(self, route):
        return self.geometries.get(
            route.route_id,
            route.world_points(self.grid_resolution),
        )


def make_route(route_id, kind, cells, source, target):
    return WFCPath(route_id, kind, source, target, tuple(cells))


class RouteDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.building = DiagnosticBuilding()

    def test_valid_slide_reports_endpoints_continuity_direction_and_slope(self):
        slide = make_route(
            "slide",
            PortKind.SLIDE,
            ((2, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 2, 0)),
            (0, 0, 2),
            (2, 0, 0),
        )

        report = diagnose_routes(self.building, [slide]).routes[0]

        self.assertEqual(report.endpoint_errors, (0.0, 0.0))
        self.assertTrue(report.continuity_valid)
        self.assertTrue(report.direction_valid)
        self.assertEqual(report.minimum_slope, 0.0)
        self.assertEqual(report.maximum_slope, float("inf"))
        self.assertFalse(report.logical_errors)

    def test_invalid_endpoints_discontinuity_and_slide_direction_are_logical_errors(self):
        slide = make_route(
            "invalid-slide",
            PortKind.SLIDE,
            ((2, 0, 0), (1, 1, 0), (3, 1, 0)),
            (0, 0, 2),
            (4, 1, 0),
        )

        report = diagnose_routes(self.building, [slide]).routes[0]

        self.assertGreater(report.endpoint_errors[1], 0)
        self.assertFalse(report.continuity_valid)
        self.assertFalse(report.direction_valid)
        self.assertEqual(len(report.logical_errors), 3)

    def test_descending_stair_is_a_logical_direction_error(self):
        stair = make_route(
            "descending-stair",
            PortKind.STAIR,
            ((2, 0, 0), (1, 0, 0), (0, 0, 0)),
            (0, 0, 0),
            (0, 0, 2),
        )

        report = diagnose_routes(self.building, [stair]).routes[0]

        self.assertFalse(report.direction_valid)
        self.assertTrue(any("stair route" in error for error in report.logical_errors))

    def test_slope_and_turn_thresholds_create_geometry_warnings(self):
        slide = make_route(
            "steep-turning-slide",
            PortKind.SLIDE,
            (
                (3, 0, 0),
                (2, 0, 0),
                (2, 1, 0),
                (1, 1, 0),
                (1, 0, 0),
                (0, 0, 0),
            ),
            (0, 0, 3),
            (0, 0, 0),
        )

        report = diagnose_routes(
            self.building,
            [slide],
            DiagnosticThresholds(
                minimum_slide_slope=0.1,
                maximum_slide_slope=0.5,
                maximum_turn_angle_degrees=10,
            ),
        ).routes[0]

        self.assertEqual(report.minimum_slope, 0.0)
        self.assertEqual(report.maximum_slope, float("inf"))
        self.assertTrue(any("slope" in warning for warning in report.warnings))
        self.assertTrue(any("turn" in warning for warning in report.warnings))
        self.assertTrue(report.problem_segments)
        self.assertIsNotNone(report.maximum_curvature)

    def test_overlapping_nonconnected_routes_are_reported_with_zero_clearance(self):
        first = make_route(
            "first",
            PortKind.SLIDE,
            ((2, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 2, 0)),
            (0, 0, 2),
            (2, 0, 0),
        )
        second = make_route(
            "second",
            PortKind.SLIDE,
            ((2, 2, 0), (1, 2, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)),
            (2, 0, 2),
            (0, 0, 0),
        )

        pair = diagnose_routes(self.building, [first, second]).route_pairs[0]

        self.assertEqual(pair.overlapping_cells, ((0, 1, 0), (1, 1, 0)))
        self.assertTrue(pair.logical_errors)
        self.assertEqual(pair.minimum_clearance, 0.0)
        self.assertTrue(pair.warnings)

    def test_nonintersecting_routes_report_spatial_clearance(self):
        first = make_route(
            "first",
            PortKind.SLIDE,
            ((2, 0, 0), (1, 0, 0), (0, 0, 0)),
            (0, 0, 2),
            (0, 0, 0),
        )
        second = make_route(
            "second",
            PortKind.SLIDE,
            ((2, 4, 0), (1, 4, 0), (0, 4, 0)),
            (4, 0, 2),
            (4, 0, 0),
        )

        pair = diagnose_routes(self.building, [first, second]).route_pairs[0]

        self.assertEqual(pair.overlapping_cells, ())
        self.assertAlmostEqual(pair.minimum_clearance, 4.0)
        self.assertFalse(pair.logical_errors)
        self.assertFalse(pair.warnings)

    def test_diagnostic_thresholds_reject_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            DiagnosticThresholds(
                minimum_slide_slope=2,
                maximum_slide_slope=1,
            )

        with self.assertRaisesRegex(ValueError, "in \\(0, 180\\]"):
            DiagnosticThresholds(maximum_turn_angle_degrees=0)

    def test_smooth_slide_reports_minimum_curvature_radius_and_violation(self):
        slide = make_route(
            "tight-curve",
            PortKind.SLIDE,
            ((2, 0, 0), (1, 0, 0), (0, 0, 0)),
            (0, 0, 2),
            (0, 0, 0),
        )
        self.building.geometries = {
            slide.route_id: tuple(
                (
                    1 + 0.1 * cos(index * 2 * pi / 32),
                    1 + 0.1 * sin(index * 2 * pi / 32),
                    1.0,
                )
                for index in range(33)
            )
        }

        report = diagnose_routes(
            self.building,
            [slide],
            DiagnosticThresholds(minimum_curvature_radius=0.2),
        ).routes[0]

        self.assertAlmostEqual(report.minimum_curvature_radius, 0.1, delta=0.002)
        self.assertTrue(any("curvature radius" in warning for warning in report.warnings))
        self.assertTrue(report.problem_segments)

    def test_straight_and_nearly_coincident_curvature_samples_are_stable(self):
        self.assertEqual(
            _curvature_radii(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
            (float("inf"),),
        )
        self.assertEqual(
            _curvature_radii(
                ((0.0, 0.0, 0.0), (1e-13, 0.0, 0.0), (1.0, 0.0, 0.0))
            ),
            (0.0,),
        )


if __name__ == "__main__":
    unittest.main()
