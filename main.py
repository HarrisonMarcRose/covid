import json
from argparse import ArgumentParser
from datetime import datetime, timedelta
from enum import Enum
from os import path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.animation import FuncAnimation, FFMpegWriter


class CovidType(Enum):
    CASES = "cases"
    VACCINATED = "margin"
    GROUPED_CASES = "cases group"
    GROUPED_DEATHS = "deaths group"
    DEATHS = "deaths"

    def __str__(self):
        return self.value


def give_me_a_straight_line(xs, ys, ws, graph:CovidType = CovidType.CASES):
    if graph != CovidType.VACCINATED:
        z = np.polyfit(xs, ys, 1, w=ws)
    else:
        z = np.polyfit(xs, ys, 1)
    f = np.poly1d(z)
    line = [f(x) for x in xs]
    return line


def save_covid_data(covid_data):
    if not path.exists('covidactnow-{}.timeseries.json'.format(
            datetime.today().strftime("%Y-%m-%d"))):
        with open('covidactnow-{}.timeseries.json'.format(
                datetime.today().strftime("%Y-%m-%d")), 'w') as file:
            return file.write(json.dumps(covid_data))

    return None


def save_election_data(election_data):
    if not path.exists('2020-election-data.json'):
        with open('2020-election-data.json', 'w') as file:
            return file.write(json.dumps(election_data))

    return None


def get_processed_election_data():
    if path.exists('2020-election-data.json'):
        with open('2020-election-data.json') as file:
            return json.loads(file.read())

    return None


def get_election_data():
    with open('precincts-with-results.geojson') as file:
        return file.read()


def get_covid_data():
    if path.exists('covidactnow-{}.timeseries.json'.format(datetime.today().strftime("%Y-%m-%d"))):
        with open('covidactnow-{}.timeseries.json'.format(datetime.today().strftime("%Y-%m-%d"))) as file:
            return json.loads(file.read())

    response = requests.get(
        url="https://api.covidactnow.org/v2/counties.timeseries.json",
        params={"apiKey": "ac42d1aa586945cb97b1e8cbaa32a02f"})

    data = []
    for county in response.json():
        data.append(
            {
                "state": county["state"],
                "county": county["county"],
                "fips": county["fips"],
                "population": county["population"],
                "data": dict()
            }

        )

        for item in county["metricsTimeseries"]:
            if item.get("vaccinationsCompletedRatio") is not None and \
                    item.get("caseDensity") is not None:
                data[-1]["data"].update({
                    item["date"]:
                        {
                            "caseDensity": item["caseDensity"],
                            "vaccinationsCompletedRatio": item["vaccinationsCompletedRatio"],
                            "vaccinationsInitiatedRatio": item["vaccinationsInitiatedRatio"]
                        }
                })

        for item in county["actualsTimeseries"]:
            if item.get("deaths") is not None:
                if data[-1]["data"].get(item["date"]):
                    data[-1]["data"][item["date"]].update(
                        {"totalDeaths": item["deaths"]})
                else:
                    data[-1]["data"].update({
                        item["date"]:
                            {"totalDeaths": item["deaths"]}
                    })

    return data


class GenAnimation:
    days = (datetime.today() - datetime(2021, 1, 1)).days
    step = 1

    def __init__(self, stream, dates, graph: CovidType):
        self.fig, self.ax = plt.subplots(figsize=(14, 7))
        self.stream = stream
        self.max_x = max(stream[-1][0])
        self.graph = graph
        self.line, self.legend, self.scat = None, None, None
        self.dems, self.reps, self.neutrals, self.unknowns = None, None, None, None

        if graph not in (CovidType.GROUPED_CASES, CovidType.GROUPED_DEATHS):
            # pick specific points to add to the legend
            x, y, s, c, w = self.stream[0]
            data = list(zip(x, y, s, c, w))
            self.dem = [point for point in data if point[3] == min(c)][0]
            self.rep = [point for point in data if point[3] == max(c)][0]
            _, mid_c = min([(abs(0.5 - color[0]), color) for color in c if color[1] == 0])
            self.neutral = [point for point in data if point[3] == mid_c][0]
            if graph != CovidType.VACCINATED:
                self.unknown = [point for point in data if point[3] == (0.5, 0.5, 0.5)][0]

        # get 5 standard deviations of y values as they have an atypical distribution
        all_ys = []
        for frame in stream:
            ys = frame[1]
            for y in ys:
                all_ys.append(y)
        all_ys.sort()
        if graph in (CovidType.CASES, CovidType.DEATHS):
            self.max_y = np.std(all_ys) * 6 + sum(all_ys) / len(all_ys)
            self.min_y = 0
            self.min_x = max(self.dem[0], self.rep[0], self.neutral[0], self.unknown[0]) + 1
        if graph in (CovidType.GROUPED_CASES, CovidType.GROUPED_DEATHS):
            self.max_y = np.std(all_ys) * 3 + sum(all_ys) / len(all_ys)
            self.min_y = max(min(all_ys), 0)
            self.min_x = 0
        if graph == CovidType.VACCINATED:
            self.max_y = max(stream[-1][1]) + 5
            self.min_y = min(stream[-1][1]) - 5
            self.min_x = max(self.dem[0], self.rep[0], self.neutral[0]) + 2

        if graph == CovidType.VACCINATED:
            self.date_text = self.ax.text(self.max_x/2, self.max_y*.9, '', fontsize=12,
                                          horizontalalignment='center')
            if graph not in (CovidType.GROUPED_CASES, CovidType.GROUPED_DEATHS):
                self.counties_text = self.ax.text(self.max_x / 3, self.max_y * .9, '', fontsize=12,
                                                  horizontalalignment='center')
        else:
            self.date_text = self.ax.text(self.max_x/2, self.max_y*.95, '', fontsize=12,
                                          horizontalalignment='center')
            if graph not in (CovidType.GROUPED_CASES, CovidType.GROUPED_DEATHS):
                self.counties_text = self.ax.text(self.max_x / 3, self.max_y * .95, '', fontsize=12,
                                                  horizontalalignment='center')

        self.dates = dates
        self.ani = FuncAnimation(self.fig,
                                 self.update,
                                 frames=GenAnimation.days // GenAnimation.step,
                                 init_func=self.setup_plot,
                                 blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c, w = self.stream[0]
        self.date_text.set_text(self.dates[0])
        if self.graph not in (CovidType.GROUPED_CASES, CovidType.GROUPED_DEATHS):
            self.counties_text.set_text("{} counties".format(len(x)))

        # x-axis label
        plt.xlabel('fully vaccinated %')

        # y-axis label
        y_labels = {
            CovidType.CASES: 'cases per 100K',
            CovidType.GROUPED_CASES: 'cases per 100K',
            CovidType.DEATHS: 'deaths per 100K',
            CovidType.GROUPED_DEATHS: 'deaths per 100K',
            CovidType.VACCINATED: '2020 election margin'
        }
        plt.ylabel(y_labels.get(self.graph))

        # plot title
        titles = {
            CovidType.CASES: 'Covid US county case vs vaccine rate '
                             '(with population and 2020 election)',
            CovidType.GROUPED_CASES: 'Covid US county case vs vaccine rate '
                                     '(with population and 2020 election)',
            CovidType.DEATHS: 'Covid US county death vs vaccine rate '
                              '(with population and 2020 election)',
            CovidType.GROUPED_DEATHS: 'Covid US county death vs vaccine rate '
                                      '(with population and 2020 election)',
            CovidType.VACCINATED: 'Covid US county vaccine rate '
                                  'vs 2020 election (size=population)'
        }
        plt.title(titles.get(self.graph))
        # showing legend

        # best line fit
        if self.graph != CovidType.VACCINATED:
            ys = give_me_a_straight_line(x, y, w, graph=self.graph)
            self.line, = self.ax.plot(x, ys, 'k:', label='best fit trend-line')

        if self.graph not in (CovidType.GROUPED_CASES, CovidType.GROUPED_DEATHS):
            # add points to legend
            self.dems = self.ax.scatter(
                [self.dem[0]], [self.dem[1]], [self.dem[2]], [self.dem[3]],
                label="pop: {}0K, dem".format(int(self.dem[2])))
            self.reps = self.ax.scatter(
                [self.rep[0]], [self.rep[1]], [self.rep[2]], [self.rep[3]],
                label="pop: {}0K, rep".format(int(self.rep[2])))
            self.neutrals = self.ax.scatter(
                [self.neutral[0]], [self.neutral[1]], [self.neutral[2]], [self.neutral[3]],
                label="pop: {}0K, neutral".format(int(self.neutral[2])))
            if self.graph != CovidType.VACCINATED:
                self.unknowns = self.ax.scatter(
                    [self.unknown[0]], [self.unknown[1]], [self.unknown[2]], [self.unknown[3]],
                    label="pop: {}0K, unknown".format(int(self.unknown[2])))
        self.ax.axis([self.min_x, self.max_x, self.min_y, self.max_y])
        self.legend = plt.legend()

        self.scat = self.ax.scatter(x, y, s=s, c=c)

        self.fig.tight_layout()

        if self.graph in (CovidType.GROUPED_CASES, CovidType.GROUPED_DEATHS):
            return self.scat, self.line, self.date_text

        if self.graph == CovidType.VACCINATED:
            return self.scat, self.date_text, self.counties_text, \
                   self.dems, self.reps, self.neutrals,

        return self.scat, self.line,  self.date_text, self.counties_text,\
            self.dems, self.reps, self.neutrals, self.unknowns

    def update(self, frame):
        x, y, s, c, w = self.stream[frame]

        self.scat.set_offsets(np.c_[x, y])
        self.scat.set_sizes(np.array(s))
        self.scat.set_color(np.array(c))

        self.date_text.set_text(self.dates[frame])
        if self.graph not in (CovidType.GROUPED_CASES, CovidType.GROUPED_DEATHS):
            self.counties_text.set_text("{} counties".format(len(x)))

        if self.graph != CovidType.VACCINATED:
            # best line fit
            ys = give_me_a_straight_line(x, y, w, graph=self.graph)
            self.line.set_xdata(x)
            self.line.set_ydata(ys)
            self.line.set_label('best fit trend-line')

        self.dems, self.reps, self.neutrals, self.unknowns = None, None, None, None

        if self.graph in (CovidType.GROUPED_CASES, CovidType.GROUPED_DEATHS):
            return self.scat, self.line, self.date_text

        if self.graph == CovidType.VACCINATED:
            return self.scat, self.date_text, self.counties_text

        return self.scat, self.line, self.date_text, self.counties_text


def add_election_info_to_covid_data(covid_data, election_data):
    # add margin info for color
    for d in covid_data:
        if election_data.get(d["fips"]):
            d["pct_dem_lead"] = (
                    (
                            election_data[d["fips"]]["votes_dem"] -
                            election_data[d["fips"]]["votes_rep"]
                    ) /
                    election_data[d["fips"]]["votes_total"]
            )
            d["color"] = (
                min(max(0.5 - d["pct_dem_lead"], 0), 1),
                0,
                max(min(0.5 + d["pct_dem_lead"], 1), 0)
            )
        else:
            d["pct_dem_lead"] = None
            d["color"] = (0.5, 0.5, 0.5)


def process_election_data(election_data):
    """Combine precints into a single county"""

    precints = [precint['properties'] for precint in election_data['features']]
    county_data = dict()
    for precint in precints:

        if precint["votes_dem"] is None \
                or precint["votes_rep"] is None \
                or precint["votes_total"] is None:
            continue

        fips = precint['GEOID'][:5]
        processed_county = county_data.get(fips)
        if processed_county is None:
            county_data[fips] = {
                "votes_dem": precint["votes_dem"],
                "votes_rep": precint["votes_rep"],
                "votes_total": precint["votes_total"],
                "precincts": 1

            }
        else:
            county_data[fips] = {
                "votes_dem": precint["votes_dem"] + county_data[fips]["votes_dem"],
                "votes_rep": precint["votes_rep"] + county_data[fips]["votes_rep"],
                "votes_total": precint["votes_total"] + county_data[fips]["votes_total"],
                "precincts": 1 + county_data[fips]["precincts"]
            }
    return county_data


def gen_plot_data(covid_data, dates, graph_type: CovidType, group=10):

    y_values = {
        CovidType.CASES: "caseDensity",
        CovidType.GROUPED_CASES: "caseDensity",
        CovidType.DEATHS: "deathDensity",
        CovidType.GROUPED_DEATHS: "deathDensity",
        CovidType.VACCINATED: "pct_dem_lead"

    }

    def data_stream():
        for date in dates:

            if graph_type != CovidType.VACCINATED:
                # x-axis values
                xs = [county["data"][date]["vaccinationsCompletedRatio"] * 100
                      for county in covid_data
                      if county
                          .get("data", {date: {"vaccinationsCompletedRatio": None}})
                          .get(date, {"vaccinationsCompletedRatio": None})
                          .get("vaccinationsCompletedRatio") is not None
                      and county
                          .get("data", {date: {y_values[graph_type]: None}})
                          .get(date, {y_values[graph_type]: None})
                          .get(y_values[graph_type]) is not None
                      ]
                # y-axis values case density
                ys = [county["data"][date][y_values[graph_type]]
                      for county in covid_data
                      if county
                          .get("data", {date: {"vaccinationsCompletedRatio": None}})
                          .get(date, {"vaccinationsCompletedRatio": None})
                          .get("vaccinationsCompletedRatio") is not None
                      and county
                          .get("data", {date: {y_values[graph_type]: None}})
                          .get(date, {y_values[graph_type]: None})
                          .get(y_values[graph_type]) is not None
                      ]
                # size
                ss = [max(1, county["population"] / 10000)
                      for county in covid_data
                      if county
                          .get("data", {date: {"vaccinationsCompletedRatio": None}})
                          .get(date, {"vaccinationsCompletedRatio": None})
                          .get("vaccinationsCompletedRatio") is not None
                      and county
                          .get("data", {date: {y_values[graph_type]: None}})
                          .get(date, {y_values[graph_type]: None})
                          .get(y_values[graph_type]) is not None
                      ]
                # weights
                ws = [county["population"]
                      for county in covid_data
                      if county
                          .get("data", {date: {"vaccinationsCompletedRatio": None}})
                          .get(date, {"vaccinationsCompletedRatio": None})
                          .get("vaccinationsCompletedRatio") is not None
                      and county
                          .get("data", {date: {y_values[graph_type]: None}})
                          .get(date, {y_values[graph_type]: None})
                          .get(y_values[graph_type]) is not None
                      ]
                cs = [county["color"]
                      for county in covid_data
                      if county
                          .get("data", {date: {"vaccinationsCompletedRatio": None}})
                          .get(date, {"vaccinationsCompletedRatio": None})
                          .get("vaccinationsCompletedRatio") is not None
                      and county
                          .get("data", {date: {y_values[graph_type]: None}})
                          .get(date, {y_values[graph_type]: None})
                          .get(y_values[graph_type]) is not None
                      ]
            else:
                xs = [county["data"][date]["vaccinationsCompletedRatio"] * 100
                      for county in covid_data
                      if county
                          .get("data", {date: {"vaccinationsCompletedRatio": None}})
                          .get(date, {"vaccinationsCompletedRatio": None})
                          .get("vaccinationsCompletedRatio") is not None
                      and county["pct_dem_lead"]]
                ys = [county["pct_dem_lead"] * 100
                      for county in covid_data
                      if county
                          .get("data", {date: {"vaccinationsCompletedRatio": None}})
                          .get(date, {"vaccinationsCompletedRatio": None})
                          .get("vaccinationsCompletedRatio") is not None
                      and county["pct_dem_lead"]]
                ss = [max(1, county["population"] / 10000)
                      for county in covid_data
                      if county
                          .get("data", {date: {"vaccinationsCompletedRatio": None}})
                          .get(date, {"vaccinationsCompletedRatio": None})
                          .get("vaccinationsCompletedRatio") is not None
                      and county["pct_dem_lead"]]
                # weights
                ws = [county["population"]
                      for county in covid_data
                      if county
                          .get("data", {date: {"vaccinationsCompletedRatio": None}})
                          .get(date, {"vaccinationsCompletedRatio": None})
                          .get("vaccinationsCompletedRatio") is not None
                      and county["pct_dem_lead"]]
                cs = [county["color"]
                      for county in covid_data
                      if county
                          .get("data", {date: {"vaccinationsCompletedRatio": None}})
                          .get(date, {"vaccinationsCompletedRatio": None})
                          .get("vaccinationsCompletedRatio") is not None
                      and county["pct_dem_lead"]]

            if graph_type in (CovidType.GROUPED_CASES, CovidType.GROUPED_DEATHS):
                chuncked_xs, chuncked_ss, chuncked_ys, chuncked_cs = [], [], [], []
                for limit in range(int(min(xs)//10),
                                   int(max([x for x in xs if x < 100])//group * group + group),
                                   group):

                    if [x for x in xs if limit <= x < limit + group]:
                        chuncked_xs.append(np.average([x for x in xs
                                                       if limit <= x < limit + group],
                                                      weights=[w for w, x in list(zip(ws, xs))
                                                               if limit <= x < limit + group]))
                        chuncked_ss.append(sum([w / 10000 for w, x in list(zip(ws, xs))
                                                if limit <= x < limit + group]))
                        chuncked_ys.append(np.average([y for y, x in list(zip(ys, xs))
                                                       if limit <= x < limit + group],
                                                      weights=[w for w, x in list(zip(ws, xs))
                                                               if limit <= x < limit + group]))
                        chuncked_cs.append(
                            (
                                np.average([c[0] for c, x in list(zip(cs, xs))
                                            if limit <= x < limit + group],
                                           weights=[w for w, x in list(zip(ws, xs))
                                                    if limit <= x < limit + group]),
                                np.average([c[1] for c, x in list(zip(cs, xs))
                                            if limit <= x < limit + group],
                                           weights=[w for w, x in list(zip(ws, xs))
                                                    if limit <= x < limit + group]),
                                np.average([c[2] for c, x in list(zip(cs, xs))
                                            if limit <= x < limit + group],
                                           weights=[w for w, x in list(zip(ws, xs))
                                                    if limit <= x < limit + group])))

                yield chuncked_xs, chuncked_ys, chuncked_ss, chuncked_cs, chuncked_ss
                continue
            yield xs, ys, ss, cs, ws

    return list(data_stream())


def fill_in_blank_dates(covid_data):
    base = datetime.today()
    date_list = [base - timedelta(days=x) for x in range(GenAnimation.days, 0, -GenAnimation.step)]
    dates = [date.strftime("%Y-%m-%d") for date in date_list]

    for county in covid_data:
        last_record, current_record = None, None
        for date in dates:
            last_record = current_record
            current_record = county["data"].get(date)
            if current_record is not None and last_record is not None:
                if current_record.get("totalDeaths") is None \
                        and last_record.get("totalDeaths") is not None:
                    county["data"][date]["totalDeaths"] = last_record["totalDeaths"]
                if current_record.get("caseDensity") is None \
                        and last_record.get("caseDensity") is not None:
                    county["data"][date]["caseDensity"] = last_record["caseDensity"]
                if current_record.get("vaccinationsCompletedRatio") is None \
                        and last_record.get("vaccinationsCompletedRatio") is not None:
                    county["data"][date]["vaccinationsCompletedRatio"] = \
                        last_record["vaccinationsCompletedRatio"]
                if current_record.get("vaccinationsInitiatedRatio") is None \
                        and last_record.get("vaccinationsInitiatedRatio") is not None:
                    county["data"][date]["vaccinationsInitiatedRatio"] = \
                        last_record["vaccinationsInitiatedRatio"]

                # Florida stopped reporting county-level deaths on June 4.
                if county["state"] == "FL" and date > "2021-06-11":
                    del county["data"][date]["totalDeaths"]

            if current_record is None and last_record is not None:
                county["data"][date] = last_record
                current_record = last_record
                if county["state"] == "FL" and date > "2021-06-11":
                    del county["data"][date]["totalDeaths"]


def calculate_death_density(covid_data):
    base = datetime.today()
    day_trend = 7
    date_list = [base - timedelta(days=x) for x
                 in range(GenAnimation.days + day_trend, 0, -GenAnimation.step)]
    dates = [date.strftime("%Y-%m-%d") for date in date_list]

    for county in covid_data:
        first_index = None
        for index, date in enumerate(dates):
            if county["data"].get(date) and county["data"][date].get("totalDeaths") \
                    and first_index is None:
                first_index = index
            if first_index is not None and index >= first_index + day_trend \
                    and county["data"][dates[index]].get("totalDeaths"):
                # doy 0 should be the minimum deaths and day 6 should be the max
                # as deaths should only go up.  However, there are corrections in the data.
                # To fix this take the max deaths - first day or last day - min deaths.
                # This ignores days after correction or days before correction and takes
                # a portion of the deaths that is the largest either before or after the correction
                total_deaths = [
                    county["data"][dates[index - days]]["totalDeaths"]
                    for days in range(day_trend)
                ]

                # if all([x > 25 for x in total_deaths]) and total_deaths != sorted(total_deaths, reverse=True):
                #     import pdb; pdb.set_trace()
                county["data"][date]["deathDensity"] = max(
                        max(total_deaths) - total_deaths[-1],  # most - first day
                        total_deaths[0] - min(total_deaths)  # last day - minimum
                    ) / day_trend * 100000 / county['population']


def select_counties(covid_data):
    top_300 = sorted([county["population"] for county in covid_data])[:1000]
    to_delete = []
    for index, county in enumerate(covid_data):
        if county["population"] not in top_300:
            to_delete.append(index)

    # in order not to change the indexes we need to delete from the end of the list first
    to_delete.sort(reverse=True)
    for delete in to_delete:
        del covid_data[delete]


def main():
    parser = ArgumentParser()
    parser.add_argument('graph', type=CovidType, choices=list(CovidType))
    parser.add_argument('--group', type=int, default=10, help="vaccinated in groups of x percent")
    opts = parser.parse_args()
    graph_type = CovidType(opts.graph)

    election_data = get_processed_election_data()
    if election_data is None:
        election_data = json.loads(get_election_data())
        election_data = process_election_data(election_data)
    save_election_data(election_data)

    covid_data = get_covid_data()
    save_covid_data(covid_data)
    fill_in_blank_dates(covid_data)
    calculate_death_density(covid_data)
    # select_counties(covid_data)

    add_election_info_to_covid_data(covid_data, election_data)
    base = datetime.today()
    date_list = [base - timedelta(days=x) for x in range(GenAnimation.days, 0, -GenAnimation.step)]
    dates = [date.strftime("%Y-%m-%d") for date in date_list]
    stream_data = gen_plot_data(covid_data, dates, graph_type, opts.group)
    animation = GenAnimation(stream_data, dates, graph_type)

    # to save an animation you need to have ffmpeg installed: brew install ffmpeg
    file_names = {
        CovidType.CASES: "covid_animation.mp4",
        CovidType.GROUPED_CASES: "covid_animation_grouped_{}.mp4".format(opts.group),
        CovidType.GROUPED_DEATHS: "covid_animation_death_grouped_{}.mp4".format(opts.group),
        CovidType.DEATHS: "covid_animation_death.mp4",
        CovidType.VACCINATED: "covid_animation_margin.mp4"
    }
    f = file_names[graph_type]
    writer_mp4 = FFMpegWriter(fps=15)
    animation.ani.save(f, writer=writer_mp4)

    plt.show()


if __name__ == "__main__":
    main()
