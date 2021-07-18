import json
from datetime import datetime, timedelta
from os import path
from statistics import mean, StatisticsError

import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.animation import FuncAnimation, FFMpegWriter


def give_me_a_straight_line(xs, ys, ws):
    w, b = np.polyfit(xs, ys, w=ws, deg=1)
    # w, b = np.polyfit(xs, ys, deg=1)  # unweighted
    line = [w * x + b for x in xs]
    return line


def save_covid_data(covid_data):
    if not path.exists('covidactnow-{}.timeseries.json'.format(datetime.today().strftime("%Y-%m-%d"))):
        with open('covidactnow-{}.timeseries.json'.format(datetime.today().strftime("%Y-%m-%d")), 'w') as file:
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

    return data


class GenAnimation:
    days = (datetime.today() - datetime(2021, 1, 1)).days
    step = 1

    def __init__(self, stream, dates):
        self.fig, self.ax = plt.subplots(figsize=(14, 7))
        self.stream = stream
        self.max_x = max(stream[-1][0])

        # get 5 standard deviations of y values as they have an atypical distribution
        all_ys = []
        for frame in stream:
            ys = frame[1]
            for y in ys:
                all_ys.append(y)
        all_ys.sort()
        self.max_y = np.std(all_ys) * 5 + sum(all_ys) / len(all_ys)
        # self.max_y = max(stream[-1][1])
        # self.min_y = min(stream[0][1])

        self.date_text = self.ax.text(self.max_x/2, self.max_y-5, '', fontsize=12, horizontalalignment='center') # 75
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

        # x-axis label
        plt.xlabel('fully vaccinated %')
        # frequency label
        plt.ylabel('cases per 100K')
        # plt.ylabel('dem vs rep county')
        # plot title
        plt.title('Covid US county case vs vaccine rate (with population and 2020 election)')
        # showing legend

        # best line fit
        ys = give_me_a_straight_line(x, y, w)
        self.line, = self.ax.plot(x, ys, 'k:', label='best fit trend-line')

        # pick specific points to add to the legend
        data = list(zip(x, y, s, c, w))
        dem = [point for point in data if point[3] == min(c)][0]
        rep = [point for point in data if point[3] == max(c)][0]
        _, mid_c = min([(abs(0.5 - color[0]), color) for color in c if color[1] == 0])
        neutral = [point for point in data if point[3] == mid_c][0]
        unknown = [point for point in data if point[3] == (0.5, 0.5, 0.5)][0]

        # add points to legend
        self.dems = self.ax.scatter([dem[0]], [dem[1]], [dem[2]], [dem[3]],
                               label="pop: {}0K, dem".format(int(dem[2]-2)))
        self.reps = self.ax.scatter([rep[0]], [rep[1]], [rep[2]], [rep[3]],
                               label="pop: {}0K, rep".format(int(rep[2]-2)))
        self.neutrals = self.ax.scatter([neutral[0]], [neutral[1]], [neutral[2]], [neutral[3]],
                                   label="pop: {}0K, neutral".format(int(neutral[2]-2)))
        self.unknowns = self.ax.scatter([unknown[0]], [unknown[1]], [unknown[2]], [unknown[3]],
                                   label="pop: {}0K, unknown".format(int(unknown[2]-2)))
        self.ax.axis([max(dem[0], rep[0], neutral[0], unknown[0]) + 1, self.max_x, 0, self.max_y])  # 80])
        self.legend = plt.legend()

        self.scat = self.ax.scatter(x, y, s=s, c=c)

        self.fig.tight_layout()
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat, self.line, self.dems, self.reps, self.neutrals, self.unknowns, self.date_text

    def update(self, frame):
        x, y, s, c, w = self.stream[frame]

        self.scat.set_offsets(np.c_[x, y])
        self.scat.set_sizes(np.array(s))
        self.scat.set_color(np.array(c))

        self.date_text.set_text(self.dates[frame])

        # best line fit
        ys = give_me_a_straight_line(x, y, w)
        self.line.set_xdata(x)
        self.line.set_ydata(ys)
        self.line.set_label('best fit trend-line')

        self.dems, self.reps, self.neutrals, self.unknowns = None, None, None, None

        return self.scat, self.line, self.date_text


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


def gen_plot_data(covid_data, dates):

    def data_stream():
        for date in dates:

            # x-axis values
            xs = [county["data"][date]["vaccinationsCompletedRatio"] * 100
                  for county in covid_data
                  if county.get("data", {date: None}).get(date)]

            # y-axis values case density
            ys = [county["data"][date]["caseDensity"]
                  for county in covid_data
                  if county.get("data", {date: None}).get(date)]

            # y-axis based on 2020 election margin
            # ys = [county["pct_dem_lead"]
            #       for county in covid_data
            #       if county.get("data", {date: None}).get(date) and county["pct_dem_lead"]]

            # size
            ss = [max(1, county["population"] / 10000) + 2
                  for county in covid_data
                  if county.get("data", {date: None}).get(date)]

            # weights
            ws = [county["population"]
                  for county in covid_data
                  if county.get("data", {date: None}).get(date)]

            cs = [county["color"]
                  for county in covid_data
                  if county.get("data", {date: None}).get(date)]

            # chuncked_xs, chuncked_ss, chuncked_ys, chuncked_cs = [], [], [], []
            # for limit in range(int(min(xs)//10), int(max([x for x in xs if x < 100])//10 * 10 +10), 10):
            #     # if date == dates[-1]:
            #     if [x for x in xs if limit <= x < limit + 10]:
            #         chuncked_xs.append(mean([x for x in xs if limit <= x < limit + 10]))
            #         chuncked_ss.append(sum([w / 10000 for w, x in list(zip(ws, xs)) if limit <= x < limit + 10]))
            #         chuncked_ys.append(np.average(
            #             [y for y, x in list(zip(ys, xs)) if limit <= x < limit + 10],
            #             weights=[w for w, x in list(zip(ws, xs)) if limit <= x < limit + 10]))
            #         chuncked_cs.append(
            #             (
            #                 np.average(
            #                     [c[0] for c, x in list(zip(cs, xs)) if limit <= x < limit + 10],
            #                     weights=[w for w, x in list(zip(ws, xs)) if limit <= x < limit + 10]),
            #                 np.average(
            #                     [c[1] for c, x in list(zip(cs, xs)) if limit <= x < limit + 10],
            #                     weights=[w for w, x in list(zip(ws, xs)) if limit <= x < limit + 10]),
            #                 np.average(
            #                     [c[2] for c, x in list(zip(cs, xs)) if limit <= x < limit + 10],
            #                     weights=[w for w, x in list(zip(ws, xs)) if limit <= x < limit + 10])))
            #
            # yield chuncked_xs, chuncked_ys, chuncked_ss, chuncked_cs, chuncked_ss
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
            if current_record is None and last_record is not None:
                county["data"][date] = last_record
                current_record = last_record


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

    election_data = get_processed_election_data()
    if election_data is None:
        election_data = json.loads(get_election_data())
        election_data = process_election_data(election_data)
    save_election_data(election_data)

    covid_data = get_covid_data()
    save_covid_data(covid_data)
    fill_in_blank_dates(covid_data)
    # select_counties(covid_data)

    add_election_info_to_covid_data(covid_data, election_data)
    base = datetime.today()
    date_list = [base - timedelta(days=x) for x in range(GenAnimation.days, 0, -GenAnimation.step)]
    dates = [date.strftime("%Y-%m-%d") for date in date_list]
    stream_data = gen_plot_data(covid_data, dates)

    # give a array of subplots for data going back in time
    # fig = plt.figure()
    # rows = 3
    # cols = 3
    # step_days = 14
    # gs = fig.add_gridspec(rows, cols, hspace=0, wspace=0, left=0.05, right=0.95, bottom=0.05, top=0.95)
    # axs = gs.subplots(sharex=True, sharey=True)
    # for i in range(rows):
    #     for j in range(cols):
    #         index = - 1 - step_days * i * cols - step_days * j
    #         axs[i, j].label_outer()
    #         axs[i, j].axis([0, 100, 0, 150])
    #         axs[i, j].scatter(
    #             stream_data[index][0],
    #             stream_data[index][1],
    #             s=stream_data[index][2],
    #             c=stream_data[index][3])
    #
    #         # best line fit
    #         ys = give_me_a_straight_line(stream_data[index][0], stream_data[index][1], stream_data[index][4])
    #         axs[i, j].plot(stream_data[index][0], ys, 'r--', label='best fit trend-line')
    #
    # plt.show()

    animation = GenAnimation(stream_data, dates)

    # to save an animation you need to have ffmpeg installed: brew install ffmpeg
    f = "covid_animation.mp4"
    writer_mp4 = FFMpegWriter(fps=15)
    animation.ani.save(f, writer=writer_mp4)

    plt.show()


if __name__ == "__main__":
    main()
