import requests
from selenium import webdriver
from time import sleep
import random
import string
from random import randrange
from secrets import username, password


class TinderBot():
    def __init__(self):
        self.driver = webdriver.Chrome()

    def cookies(self):
        ck_btn = self.driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[2]/div/div/div[1]/button')
        ck_btn.click()

    def login(self):
        self.driver.get('https://tinder.com')

        sleep(4)
        self.cookies()
        sleep(2)

        fb_btn = self.driver.find_element_by_xpath(
            '//*[@id="modal-manager"]/div/div/div[1]/div/div[3]/span/div[2]/button'
        )
        fb_btn.click()

        base_window = self.driver.window_handles[0]
        self.driver.switch_to_window(self.driver.window_handles[1])

        email_in = self.driver.find_element_by_xpath('//*[@id="email"]')
        email_in.send_keys(username)

        pw_in = self.driver.find_element_by_xpath('//*[@id="pass"]')
        pw_in.send_keys(password)

        login_btn = self.driver.find_element_by_xpath('//*[@id="u_0_0"]')
        login_btn.click()

        self.driver.switch_to_window(base_window)

        sleep(10)

        popup_1 = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div/div/div[3]/button[1]')
        popup_1.click()

        sleep(10)

        popup_2 = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div/div/div[3]/button[1]')
        popup_2.click()

    def like(self):
        like_btn = self.driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[1]/div/main/div[1]/div/div/div[1]/div/div[2]/div[4]/button'
        )
        like_btn.click()

    def dislike(self):
        dislike_btn = self.driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[1]/div/main/div[1]/div/div/div[1]/div/div[2]/div[2]/button'
        )
        dislike_btn.click()

    def get_random_string(self):
        result = ''.join(random.choice(string.ascii_uppercase) for x in range(10))
        print(result)
        return result

    def take_screenshot(self):
        filename = self.get_random_string() + '.png'
        image = self.driver.find_element_by_xpath(
            '//*[@id="content"]/div/div[1]/div/main/div[1]/div/div/div[1]/div/div[1]/div[3]/div[1]/div/span/div')
        image_url = image.value_of_css_property("background-image")
        raw_url = image_url.replace('url("', '').replace('")', '')
        # get webp
        # convert webp --> png
        resp = requests.get(raw_url)
        im = Image.open(BytesIO(resp.content)).convert("RGB")
        im.save(filename, "png")

    def auto_swipe(self):
        irand = randrange(2, 5)
        left, right = 0, 0
        while True:
            sleep(1)
            try:
                rand = random.uniform(0, 1)
                if rand > .1:
                    sleep(irand)
                    self.like()
                    self.take_screenshot()
                    right += 1
                    print('{}th right swipe'.format(right))
                else:
                    self.dislike()
                    sleep(irand)
                    left += 1
                    print('{}th left swipe'.format(left))
            except Exception:
                try:
                    self.close_popup()
                except Exception:
                    self.close_match()

    def close_popup(self):
        popup_3 = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div[2]/button[2]')
        popup_3.click()

    def close_match(self):
        match_popup = self.driver.find_element_by_xpath('//*[@id="modal-manager-canvas"]/div/div/div[1]/div/div[3]/a')
        match_popup.click()

    def message_all(self):
        irand = randrange(2, 5)
        while True:
            matches = self.driver.find_elements_by_class_name('matchListItem')

            sleep(irand)

            if len(matches) < 2:
                break

            matches[1].click()
            msg_box = self.driver.find_element_by_class_name('sendMessageForm__input')
            rand = random.uniform(0, 1)

            msg_box.send_keys('hola') if rand > .5 else msg_box.send_keys('hello (:')
            send_btn = self.driver.find_element_by_xpath(
                '//*[@id="content"]/div/div[1]/div/main/div[1]/div/div/div/div[1]/div/div/div[3]/form/button')
            send_btn.click()

            sleep(irand)

            matches_tab = self.driver.find_element_by_xpath('//*[@id="match-tab"]')
            matches_tab.click()

            sleep(irand)


bot = TinderBot()
bot.login()
bot.auto_swipe()
# bot.message_all()
