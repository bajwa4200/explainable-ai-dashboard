from xai.model import build_report, train_classifier


def test_train_report_accuracy():
    model, X_test, y_test, names = train_classifier(random_state=0)
    report = build_report(model, X_test, y_test, names)
    assert report.accuracy > 0.8
    assert len(report.importances) == len(names)
