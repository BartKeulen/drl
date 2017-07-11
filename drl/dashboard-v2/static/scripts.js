/**
 * Created by bartkeulen on 7/6/17.
 */
var UPDATE_INTERVAL = 500000;
var MA_PARAM = 5;
var interval = null;

$(document).ready(function() {
    // Set active session cookie
    if (!Cookies.get().hasOwnProperty("active_sessions")) {
        Cookies.set("active_sessions", JSON.stringify([]));
    }

    // Set moving average parameter cookie
    if (!Cookies.get().hasOwnProperty("ma_param")) {
        Cookies.set("ma_param", MA_PARAM);
    }

    // Set update interval parameter cookie
    if (!Cookies.get().hasOwnProperty("update_param")) {
        Cookies.set("update_param", UPDATE_INTERVAL);
    }

    // Initialize view
    set_view();

    // On hash change set correct view
    window.onhashchange = function() {
        set_view();
    };

    // Set path cookie
    $("#path").on("input", function(input) {
        Cookies.set("directory", input.currentTarget.value);

        if (window.location.hash.substr(1) === "sessions") {
            set_view();
        }
    });

    $(document).on("click", ".row-clickable", function(event) {
        if (!$(event.target).hasClass("check-session")) {
            if ($(this).next().hasClass("runs-hidden")) {
                $(this).nextUntil(".row-clickable").removeClass("runs-hidden").addClass("runs-visible");
            } else {
                $(this).nextUntil(".row-clickable").removeClass("runs-visible").addClass("runs-hidden");

            }
        }
    });

    $(document).on("change", ".check-full-session", function() {
        $(this).parent().parent().nextUntil(".row-clickable").find(".check-run").prop("checked", this.checked);
        update_active_sessions();
    });

    $(document).on("change", ".check-run", function() {
        update_active_sessions();
    });

    $(document).on("change", ".check-average", function() {
        update_average_sessions();
    });

    $(document).on("change", "#ma-param", function() {
        Cookies.set("ma_param", this.value);
    });

    $(document).on("change", "#update-param", function() {
        Cookies.set("update_param", this.value);
    });
});

function set_view() {
    if (interval != null) {
        clearInterval(interval);
    }
    var anchor = document.location.hash.substr(1);
    if (anchor === "sessions") {
        get_summaries(function(data) {
            render_sessions(data);
        });
    }
    else if (anchor === "filters") {
        var data = {
            "ma-param": Cookies.get("ma_param"),
            "update-param": Cookies.get("update_param")
        };
        render_filters(data);
    }
    else {
        get_active_sessions(function(data) {
            render_graphs(data[0], function() {
                update_chart_data(data);
                interval = setInterval(chart_update_loop, Cookies.get("update_param"));
            });
        });
    }
}

function render_graphs(chart_ids, cb) {
    $.get("static/graphs.hbs", function(source) {
        var template = Handlebars.compile(source);
        var html = template(chart_ids);
        $("#page-title").empty().append("Graphs");
        $('#main-content').empty().append(html);
        cb();
    });
}

function render_sessions(data) {
    $.get("static/sessions.hbs", function(source) {
        var template = Handlebars.compile(source);
        var html = template(data);
        $("#page-title").empty().append("Sessions");
        $('#main-content').empty().append(html);
    });
}

function render_filters(data) {
    $.get("static/filters.hbs", function(source) {
        var template = Handlebars.compile(source);
        var html = template(data);
        $("#page-title").empty().append("Filters");
        $('#main-content').empty().append(html);
    });
}

// Get all summaries
function get_summaries(cb) {
    $.getJSON("/summaries", function (data) {
        cb(data);
    });
}

function update_active_sessions() {
    var paths = [];
    var count = $(".check-run").length;
    $(".check-run").each(function (idx) {
        if (this.checked) {
            paths.push(this.id);
        }
        if (!--count) Cookies.set("active_sessions", JSON.stringify(paths));
    });
}

function update_average_sessions() {
    var paths = [];
    var count = $(".check-average").length;
    $(".check-average").each(function(idx) {
        if (this.checked) {
            paths.push(this.id);
        }
        if (!--count) Cookies.set("average_sessions", JSON.stringify(paths));
    });
}

function get_active_sessions(cb) {
    var active = JSON.parse(Cookies.get("active_sessions"));
    var average = JSON.parse(Cookies.get("average_sessions"));
    $.ajax({
        type: "POST",
        url: "/active_sessions",
        data:  JSON.stringify({active: active, average: average}),
        contentType: "application/json;charset=UTF-8",
        dataType: "json",
        success: function(response, status, jqXHR) {
            cb(convert_data_to_chart(response));
        },
        error: function(response, status, error) {
            alert("Error: " + error + ". Status: " + status);
            cb([]);
        }
    });
}

function convert_data_to_chart(data) {
    var charts = {};
    var charts_average_tmp = {};
    for (var i in data) {
        var env = data[i][2]["info"]["env"];
        if (!charts.hasOwnProperty(env)) {
            charts[env] = [];
            charts_average_tmp[env] = {};
        }

        var value_types = Object.keys(data[i][1]["values"]);
        for (var j in value_types) {
            var id = get_chart_id(env, value_types[j]);
            var x = data[i][1]["episode"];
            var y = data[i][1]["values"][value_types[j]];

            if (data[i][3]) {
                var name = data[i][2]["info"]["name"] + " - " + data[i][2]["info"]["timestamp"] + " - " + data[i][2]["info"]["run"];

                var filt = moving_average(x, y);
                var values = {
                    x: filt[0],
                    y: filt[1],
                    name: name,
                    mode: 'lines',
                    line: {width: 1}
                };

                var exists = false;
                for (var k in charts[env]) {
                    if (charts[env][k]["id"] === id) {
                        charts[env][k]["values"].push(values);
                        exists = true;
                    }
                }
                if (!exists) {
                    charts[env].push({"id": id, "values": [values]});
                }
            }
            else {
                var name = data[i][2]["info"]["name"] + " - " + data[i][2]["info"]["timestamp"];
                var name_tag = name.split(" ").join("");
                if (!charts_average_tmp[env].hasOwnProperty(name_tag)) {
                    charts_average_tmp[env][name_tag] = [];
                }

                var exists = false;
                for (var k in charts_average_tmp[env][name_tag]) {
                    if (charts_average_tmp[env][name_tag][k]["id"] === id) {
                        charts_average_tmp[env][name_tag][k]["tmp"].push({x: x, y: y});
                        exists = true;
                    }
                }
                if (!exists) {
                    charts_average_tmp[env][name_tag].push({"id": id, name: name, "tmp": [{x: x, y: y}]});
                }
            }
        }
    }

    var charts_average = {};
    for (var env in charts_average_tmp) {
        charts_average[env] = [];
        for (var name_tag in charts_average_tmp[env]) {
            for (var i in charts_average_tmp[env][name_tag]) {
                var id = charts_average_tmp[env][name_tag][i]["id"];
                var name = charts_average_tmp[env][name_tag][i]["name"];
                var chart_data = average_sessions(charts_average_tmp[env][name_tag][i]["tmp"]);

                var values = {
                    x: chart_data["x"],
                    y: chart_data["y"],
                    name: name,
                    mode: 'lines',
                    line: {width: 1}
                };

                var exists = false;
                for (var k in charts[env]) {
                    if (charts[env][k]["id"] === id) {
                        charts[env][k]["values"].push(values);
                        exists = true;
                    }
                }
                if (!exists) {
                    charts[env].push({"id": id, "values": [values]});
                }
            }
        }
    }

    console.log(charts);

    return [charts, charts_average];
}

function update_chart_data(data) {
    var charts = data[0];
    for (var env in charts) {
        for (var i in charts[env]) {
            var elem = document.getElementById(charts[env][i]["id"]);

            var values = charts[env][i]["values"];

            var layout = {
                title: charts[env][i]["id"],
                showlegend: true,
                legend: {"orientation": "h"}
            };

            Plotly.newPlot(elem, values, layout);
        }
    }
}

function chart_update_loop() {
    get_active_sessions(function(charts) {
        for (var env in charts) {
            for (var i in charts[env]) {
                var elem = document.getElementById(charts[env][i]["id"]);
                elem.data = charts[env][i]["values"];
                Plotly.redraw(elem);
            }
        }
    });
}

function get_chart_id(env, value_type) {
    return (env + "_" + value_type).split(" ").join("-");
}

function moving_average(episode, data) {
    var new_episode = [];
    var new_data = [];
    var n = Cookies.get("ma_param");
    for (var i = 0; i < data.length-n+1; i++) {
        var sum = 0.;
        for (var j = 0; j < n; j++) {
            sum += data[i+j];
        }
        new_data.push(sum/n);
        new_episode.push(episode[i])
    }
    return [new_episode, new_data];
}

function average_sessions(data) {
    var x = [];
    var y = [];
    for (var i in data) {
        if (i == 0) {
            x = data[i]["x"];
            y = data[i]["y"];
        }
        else {
            if (x.length !== data[i]["x"].length || y.length !== data[i]["y"].length) {
                return {x: x, y: y};
            }

            for (var j=0; j<x.length; j++) {
                y[j] += data[i]["y"][j];
            }
        }

        for (var j=0; j<x.length; j++) {
            y[j] /= data[i]["y"].length;
        }
    }
    return {x: x, y: y};
}