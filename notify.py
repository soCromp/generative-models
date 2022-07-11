import smtplib, ssl
import sys

class email:
    def __init__(self, subject='', message='') -> None:
        context = ssl.create_default_context()
        port = 465  # For SSL
        password = 'wqhobbatxvnoexnl' # THIS IS A THROWAWAY ACCOUNT! Nothing interesting here if you get into it

        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login('mycodeisdone@gmail.com', password)
            server.sendmail('mycodeisdone@gmail.com', 'mycodeisdone@gmail.com', f'Subject: {subject}\n\n{message}')

if __name__ == '__main__':
    try: content = '' + sys.argv[-1]
    except: content = '    '
    subject = 'The code is done!'
    email(subject, content)
        