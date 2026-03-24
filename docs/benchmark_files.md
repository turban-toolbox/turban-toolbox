# Test and benchmark data files

TURBAN comes with an extensive set of unit tests and example code
snippets. The data files used by these tests and examples are not
included in the git repository, but are stored on a server. For these
tests to work, the user would need to download these files. 

There are two routes to do this:

* manual download
* automatic download

In either case, files are downloaded only if one or more data files appear
to be missing. Note that some of these files are large and downloading
can take some time, depending on the bandwidth of your internet
connection.

The manual download is default and requires explicit user action to
initiate the download. This can be achieved by
```bash
python tests/filepaths.py
```

The downloading procedure can be made automatically by setting the
environment variable ```TURBAN_AUTO_DOWNLOAD_FILES```, like so
```bash
export TURBAN_AUTO_DOWNLOAD_FILES=1
```
Whenever a unit test is run, and one or more data files appear to be
missing, the data files are downloaded automatically. This feature is
intended primarily for developers.


