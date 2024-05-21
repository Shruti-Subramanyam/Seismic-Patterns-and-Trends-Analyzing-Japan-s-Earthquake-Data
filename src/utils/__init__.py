def gen_year_month_list(start_year, start_month, end_year, end_month):
    year = [str(y) for y in range(start_year, end_year + 1)]
    month = [f"{i:02}" for i in range(1, 13)]
    year_month = [y + m for y in year for m in month]
    year_month_sub = list(
        filter(lambda x: int(f"{start_year}{start_month:02}") <= int(x) <= int(f"{end_year}{end_month:02}"),
               year_month))
    return year_month_sub
