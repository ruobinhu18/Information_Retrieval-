from typing import Dict, Union, Iterator
import functools
import os
import time
import re


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_t = time.perf_counter()
        f_value = func(*args, **kwargs)
        elapsed_t = time.perf_counter() - start_t
        mins = elapsed_t // 60
        print(
            f"'{func.__name__}' elapsed time: {mins} minutes, {elapsed_t - mins * 60:0.2f} seconds"
        )
        return f_value

    return wrapper_timer


def load_wapo(wapo_jl_path: Union[str, os.PathLike]) -> Iterator[Dict]:
    """
    It should be similar to the load_wapo in HW3 with two changes:
    - for each yielded document dict, use "doc_id" instead of "id" as the key to store the document id.
    - convert the value of "published_date" to a readable format e.g. 2021/3/15. You may consider using python datetime package to do this.
    """
    lst = []
    idx = -1
    with open(wapo_jl_path, "r") as file:
        for line in file:
            idx += 1
            # convert str to dict
            line = line.replace('null', 'None')
            line = line.replace('true', "'true'")
            line = line.replace('false', "'false'")
            line = eval(line)
            # generate content_str
            contents = line["contents"]
            content_str = ""
            for d in contents:
                if d is None:
                    pass
                elif d['type'] == 'sanitized_html':
                    content_str += d['content'] + " "
            content_str.strip()
            # handle none title
            if line['title'] is None:
                line['title'] = content_str[:70]
            # remove html tags
            line['title'] = re.sub(r'<[^>]*>', ' ', line['title'])
            content_str = re.sub(r'<[^>]*>', ' ', content_str)
            # convert published_date
            t = line['published_date'] / 1000
            date = time.strftime('%Y/%m/%d', time.localtime(t))
            # generate dict
            lst.append({"doc_id": idx, "title": line["title"], "author": line["author"],
                        "published_date": date, "content_str": content_str})
    return iter(lst)


if __name__ == "__main__":
    pass
